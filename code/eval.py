"""
测试 Base / SFT checkpoint（本地运行）

Base 模型（纯文本续写）：
    python eval.py --latest
    python eval.py --checkpoint step_020000.pt
    python eval.py --latest --perplexity

SFT 模型（对话格式）：
    python eval.py --latest --sft                                    # 自动找最新 SFT checkpoint
    python eval.py --checkpoint-dir checkpoints_sft_v31 --step 5000 --sft
    python eval.py --latest --sft --temp 0.7 --top-p 0.8

Stage 2 身份注入模型（对话格式）：
    python eval.py --checkpoint-dir checkpoints_sft2_v51 --sft2 --step 1000 --sft --temp 0.6 --top-p 0.9 --top-k 40
    python eval.py --checkpoint-dir checkpoints_sft2_v51 --sft2 --latest --sft
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from pathlib import Path

# ============================================================
BASE_DIR = Path(__file__).parent
TOKENIZER_PATH = BASE_DIR / "tokenizer_minimind_8k" / "tokenizer.json"

# 词表从分词器自动读取（不再硬编码）
from tokenizers import Tokenizer as TokReader
_tok = TokReader.from_file(str(TOKENIZER_PATH))
VOCAB_SIZE = _tok.get_vocab_size()

CONFIG = {
    "vocab_size": VOCAB_SIZE,  # 自动匹配分词器（8K ~13K tokens）
    "n_layer": 20,
    "n_head": 16,
    "n_query_groups": 4,       # v29c/v28：GQA 4 KV 组
    "n_embd": 1024,
    "intermediate_size": 3584,
    "block_size": 1024,
    "norm_eps": 1e-5,
    "diff_attention": False,   # StdAttn
    "qk_norm": True,           # v29c/v28：开 QK-Norm
}

# ============================================================
# 模型定义（与训练完全一致）
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def apply_rope(x, cos, sin):
    # cos/sin: [1, 1, T, hd//2]，x: [B, nh, T, hd]
    # 直接 broadcast，不需要额外 unsqueeze（v31 fix）
    r = x.float().reshape(*x.shape[:-1], -1, 2)
    out0 = r[..., 0] * cos - r[..., 1] * sin
    out1 = r[..., 1] * cos + r[..., 0] * sin
    return torch.stack([out0, out1], dim=-1).flatten(-2).to(x.dtype)


class StdAttn(nn.Module):
    """v23-d：标准多头注意力（替换 DiffAttn，排查 softmax 相减梯度抑制）"""
    def __init__(self, c):
        super().__init__()
        self.nh = c["n_head"]
        self.hd = c["n_embd"] // c["n_head"]
        self.ng = c.get("n_query_groups", self.nh)
        self.scale = self.hd ** -0.5
        kv_dim = self.hd * self.ng
        self.q_proj = nn.Linear(c["n_embd"], c["n_embd"], bias=False)
        self.k_proj = nn.Linear(c["n_embd"], kv_dim, bias=False)
        self.v_proj = nn.Linear(c["n_embd"], kv_dim, bias=False)
        self.o_proj = nn.Linear(c["n_embd"], c["n_embd"], bias=False)
        if c.get("qk_norm", False):
            self.q_norm = RMSNorm(self.hd)
            self.k_norm = RMSNorm(self.hd)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x, cos, sin, mask=None):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd)
        k = self.k_proj(x).view(B, T, self.ng, self.hd)
        v = self.v_proj(x).view(B, T, self.ng, self.hd)
        rp = self.nh // self.ng
        if rp > 1:
            k = k.repeat_interleave(rp, dim=2)
            v = v.repeat_interleave(rp, dim=2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # ★ SDPA Flash Attention — 与训练 notebook 完全一致
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c["n_embd"]; i = c["intermediate_size"]
        self.w1 = nn.Linear(d, i, bias=False)
        self.w2 = nn.Linear(d, i, bias=False)
        self.w3 = nn.Linear(i, d, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.attn_norm = RMSNorm(c["n_embd"], c["norm_eps"])
        self.attn = StdAttn(c)  # v23-d: 标准注意力
        self.ffn_norm = RMSNorm(c["n_embd"], c["norm_eps"])
        self.ffn = SwiGLU(c)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Shayler(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cfg = c; self.bs = c["block_size"]
        self.emb = nn.Embedding(c["vocab_size"], c["n_embd"])
        self.layers = nn.ModuleList([Block(c) for _ in range(c["n_layer"])])
        self.norm = RMSNorm(c["n_embd"], c["norm_eps"])
        self.head = nn.Linear(c["n_embd"], c["vocab_size"], bias=True)  # v23
        self.emb.weight = self.head.weight
        freqs = 1.0 / (10000 ** (torch.arange(0, c["n_embd"] // c["n_head"], 2).float() / (c["n_embd"] // c["n_head"])))
        t = torch.arange(self.bs).float()
        a = torch.outer(t, freqs)
        self.register_buffer("rc", torch.cos(a).unsqueeze(0).unsqueeze(0))
        self.register_buffer("rs", torch.sin(a).unsqueeze(0).unsqueeze(0))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.emb(idx)
        cos = self.rc[:, :, :T].to(x.device)
        sin = self.rs[:, :, :T].to(x.device)
        # ★ SDPA is_causal 内部处理 causal mask，无需手动构建
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.head(x)
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, self.cfg["vocab_size"]), targets.view(-1))
            return ce
        return logits


# ============================================================
# 续写函数
# ============================================================

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new=64, temperature=0.8, top_p=0.9,
             top_k=50, repetition_penalty=1.1):
    """纯文本续写"""
    model.eval()
    device = next(model.parameters()).device

    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generated = []

    for _ in range(max_new):
        if input_ids.shape[1] > 1024:
            input_ids = input_ids[:, -1024:]

        logits = model(input_ids)
        next_logits = logits[:, -1, :].float()

        # 重复惩罚：对已生成的 token 降权
        if repetition_penalty != 1.0 and len(generated) > 0:
            for token_id in set(generated):
                if next_logits[0, token_id] > 0:
                    next_logits[0, token_id] /= repetition_penalty
                else:
                    next_logits[0, token_id] *= repetition_penalty

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_k > 0:
            top_k_values, top_k_indices = torch.topk(next_logits, min(top_k, next_logits.shape[-1]))
            mask = torch.full_like(next_logits, float('-inf'))
            mask.scatter_(1, top_k_indices, top_k_values)
            next_logits = mask

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
            cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_logits[indices_to_remove] = float('-inf')

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(generated)


# ============================================================
# 困惑度评估（Base 模型核心指标）
# ============================================================

@torch.no_grad()
def eval_perplexity(model, tokenizer, data_files, max_samples=200):
    """在训练数据上评估困惑度（v31：2D .pt 文件 [N, 1024]）"""
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    for fp in data_files[:max_samples]:
        try:
            ids = torch.load(fp, map_location="cpu", weights_only=True).long()
        except Exception:
            continue

        # v31 数据是 2D [N, 1024]，展平为 1D
        if ids.dim() == 2:
            ids = ids.view(-1)

        # 取 block_size+1 长度的切片，stride=block_size//2（50% 重叠）
        for i in range(0, len(ids) - CONFIG["block_size"] - 1, CONFIG["block_size"] // 2):
            chunk = ids[i:i + CONFIG["block_size"] + 1]
            if len(chunk) < 128:
                break
            x = chunk[:-1].unsqueeze(0).to(device)
            y = chunk[1:].unsqueeze(0).to(device)

            loss = model(x, y)
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()

            if total_tokens >= max_samples * CONFIG["block_size"]:
                break

        if total_tokens >= max_samples * CONFIG["block_size"]:
            break

    if total_tokens == 0:
        return None, None

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="测试 Base 预训练 checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径")
    parser.add_argument("--latest", action="store_true", help="自动使用最新 checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_sft_v31", help="checkpoint 目录")
    parser.add_argument("--step", type=int, default=None, help="指定 step 的 checkpoint")
    parser.add_argument("--temp", type=float, default=0.8, help="温度 (默认 0.8)")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling (默认 0.9)")
    parser.add_argument("--top-k", type=int, default=50, help="top-k 采样 (默认 50)")
    parser.add_argument("--rep-penalty", type=float, default=1.1, help="重复惩罚 (默认 1.1)")
    parser.add_argument("--perplexity", action="store_true", help="评估困惑度（需要 tokenized 数据）")
    parser.add_argument("--sft", action="store_true", help="SFT Stage 1 对话格式测试（使用 user：/ assistant：模板）")
    parser.add_argument("--sft2", action="store_true", help="SFT Stage 2 身份注入 checkpoint（前缀 sft2_step_）")
    parser.add_argument("--data-dir", type=str, default=None, help="tokenized 数据目录（默认 pretrain_tokenized_8k_v30）")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # --- 加载分词器 ---
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    print(f"词表: {tokenizer.get_vocab_size()}")

    # --- 解析 checkpoint ---
    ckpt_dir = Path(args.checkpoint_dir)

    # Checkpoint 前缀：sft2_step_ (Stage 2) > sft_step_ (Stage 1) > step_ (Base)
    if args.sft2:
        ckpt_prefix = "sft2_step_"
    elif args.sft:
        ckpt_prefix = "sft_step_"
    else:
        ckpt_prefix = "step_"

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    elif args.step is not None:
        ckpt_path = ckpt_dir / f"{ckpt_prefix}{args.step:06d}.pt"
        if not ckpt_path.exists():
            # 尝试另一个前缀
            alt_prefix = "step_" if args.sft else "sft_step_"
            ckpt_path = ckpt_dir / f"{alt_prefix}{args.step:06d}.pt"
    elif args.latest:
        ckpts = sorted(ckpt_dir.glob(f"{ckpt_prefix}*.pt"))
        if not ckpts:
            # 回退：尝试另一种前缀
            alt_prefix = "step_" if args.sft else "sft_step_"
            ckpts = sorted(ckpt_dir.glob(f"{alt_prefix}*.pt"))
        if not ckpts:
            print(f"[错误] {ckpt_dir} 中没有 checkpoint")
            return
        ckpt_path = ckpts[-1]
    else:
        ckpts = sorted(ckpt_dir.glob(f"{ckpt_prefix}*.pt"))
        if not ckpts:
            alt_prefix = "step_" if args.sft else "sft_step_"
            ckpts = sorted(ckpt_dir.glob(f"{alt_prefix}*.pt"))
        if ckpts:
            ckpt_path = ckpts[-1]
            print(f"📂 自动选择最新: {ckpt_path.name}")
        else:
            print(f"[错误] 未指定 checkpoint，且 {ckpt_dir} 中无 checkpoint")
            print(f"用法: python eval.py --latest  或  python eval.py --checkpoint xxx.pt")
            return

    print(f"\n加载: {ckpt_path.name}  ({ckpt_path.stat().st_size / 1e6:.0f} MB)")
    sd = torch.load(ckpt_path, map_location="cpu")

    # --- checkpoint 信息 ---
    print(f"\n{'='*50}")
    print(f"  Checkpoint 信息")
    print(f"{'='*50}")
    ckpt_vocab = sd["model"]["emb.weight"].shape[0]
    print(f"  词表大小: {ckpt_vocab}")
    CONFIG["vocab_size"] = ckpt_vocab

    step = sd.get("step", "?")
    loss_ema = sd.get("loss_ema", None)
    print(f"  训练步数: {step}")
    if loss_ema is not None:
        print(f"  Loss EMA:  {loss_ema:.4f}")
    if "config" in sd:
        ckpt_cfg = sd["config"]
        print(f"  层数: {ckpt_cfg.get('n_layer', '?')}")
        print(f"  维度: {ckpt_cfg.get('n_embd', '?')}")
        print(f"  batch: {ckpt_cfg.get('micro_batch_size', '?')}")

    # --- 加载模型 ---
    print(f"\n创建模型...")
    model = Shayler(CONFIG)
    model.load_state_dict(sd["model"])
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"参数: {params:.1f}M")

    # --- 形状校验 ---
    print(f"\n{'='*50}")
    print(f"  形状校验")
    print(f"{'='*50}")
    expected = {
        "emb.weight": (CONFIG["vocab_size"], CONFIG["n_embd"]),
        "head.weight": (CONFIG["vocab_size"], CONFIG["n_embd"]),
        "layers.0.attn.q_proj.weight": (CONFIG["n_embd"], CONFIG["n_embd"]),
        "layers.0.ffn.w3.weight": (CONFIG["n_embd"], CONFIG["intermediate_size"]),
        "layers.19.ffn.w3.weight": (CONFIG["n_embd"], CONFIG["intermediate_size"]),
        "norm.weight": (CONFIG["n_embd"],),
    }
    all_ok = True
    for name, shape in expected.items():
        actual = sd["model"][name].shape
        ok = "✅" if actual == shape else "❌"
        if actual != shape:
            all_ok = False
        print(f"  {ok} {name}: {list(actual)}")

    # --- 困惑度 ---
    if args.perplexity:
        data_dir = args.data_dir or str(BASE_DIR / "base_data" / "pretrain_pt")
        data_path = Path(data_dir)
        if data_path.exists():
            files = sorted(data_path.glob("*.pt"))
            print(f"\n{'='*50}")
            print(f"  困惑度评估 ({len(files)} 个数据文件)")
            print(f"{'='*50}")
            avg_loss, ppl = eval_perplexity(model, tokenizer, files)
            if ppl is not None:
                print(f"  Loss:      {avg_loss:.4f}")
                print(f"  Perplexity: {ppl:.1f}")
            else:
                print(f"  [跳过] 无有效数据")
        else:
            print(f"\n[跳过困惑度] 数据目录不存在: {data_dir}")

    # --- 续写测试 ---
    print(f"\n{'='*50}")
    print(f"  续写测试 (temp={args.temp}, top_p={args.top_p}, top_k={args.top_k}, rep_penalty={args.rep_penalty})")
    print(f"{'='*50}")

    if args.sft:
        # SFT 对话格式测试（模型学过 user：/ assistant： 模板）
        tests = [
            ("问候", "user：你好呀，今天心情怎么样？\nassistant："),
            ("知识", "user：光合作用是什么？\nassistant："),
            ("闲聊", "user：我最近有点累，想放松一下\nassistant："),
            ("创作", "user：写一首关于夏天的小诗\nassistant："),
            ("逻辑", "user：为什么天空是蓝色的？\nassistant："),
            ("角色", "user：你是谁？你叫什么名字？\nassistant："),
            ("建议", "user：有什么保持健康的好习惯吗？\nassistant："),
            ("多轮", "user：你好呀\nassistant：你好！今天过得怎么样？\nuser：还不错，刚跑完步\nassistant："),
        ]
    else:
        tests = [
            ("常识", "今天天气真好，"),
            ("百科", "中华人民共和国成立于"),
            ("科技", "机器学习是人工智能的一个分支，"),
            ("叙事", "从前有一座山，山上"),
            ("数学", "计算圆的面积公式是"),
            ("科普", "光合作用是植物利用"),
            ("古文", "子曰：学而时习之，"),
            ("问答", "如何保持健康的身体？"),
        ]

    for label, prompt in tests:
        out = generate(model, tokenizer, prompt, max_new=80,
                       temperature=args.temp, top_p=args.top_p,
                       top_k=args.top_k, repetition_penalty=args.rep_penalty)
        # 截断显示
        display = out[:120].replace("\n", "\\n")
        print(f"\n  [{label}]")
        print(f"  {prompt}{display}")

    print(f"\n{'='*50}")
    print(f"  交互续写（输入 quit 退出）")
    print(f"{'='*50}")

    while True:
        try:
            user = input("\nprompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见~")
            break
        if not user:
            continue
        if user.lower() == "quit":
            break
        out = generate(model, tokenizer, user, max_new=128,
                       temperature=args.temp, top_p=args.top_p,
                       top_k=args.top_k, repetition_penalty=args.rep_penalty)
        print(f"{user}{out}")


if __name__ == "__main__":
    main()
