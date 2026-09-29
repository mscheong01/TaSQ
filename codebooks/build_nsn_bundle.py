"""Package NSNQuant's released 1-bit codebook into a bundle the serving fork's `nsn_quant.py`
hook can load. The codebook is global and calibration-free -- one synthetic 256x8 table shared by
every layer, head and side -- so unlike build_cq_bundle.py this reads no calibration pickle.
"""
import argparse

import torch
from transformers import AutoConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


def _pseudo_quantize_tensor_4bit(w: torch.Tensor, group_size: int) -> torch.Tensor:
    shape = w.shape
    flat = w.reshape(-1, group_size)
    w_max = flat.amax(dim=-1, keepdim=True)
    w_min = flat.amin(dim=-1, keepdim=True)
    scale = torch.clamp((w_max - w_min) / 15.0, min=1e-5)
    q = ((flat - w_min) / scale).clamp_(0, 15).round_()
    return (q * scale + w_min).reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook_path", default="NSNQuant/codebooks/1bit_codebook.pt")
    ap.add_argument("--model_name", default="NousResearch/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--window_size", type=int, default=64)
    ap.add_argument("--out", required=True, help="output bundle .pt path")
    args = ap.parse_args()

    codebook = torch.load(args.codebook_path, weights_only=True)
    assert codebook.shape[-1] == 8, f"expected 8-dim VQ groups, got {codebook.shape}"
    assert codebook.shape[0] == 256, f"expected 256-entry (1-bit) codebook, got {codebook.shape}"
    codebook = _pseudo_quantize_tensor_4bit(codebook.float(), group_size=8)

    cfg = AutoConfig.from_pretrained(args.model_name)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    rope_scaling = getattr(cfg, "rope_scaling", None) or {}
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    # "default"/None both mean "no actual scaling" in HF's rope_scaling schema -- ROPE_INIT_FUNCTIONS
    # has no "default" entry (only real scaling strategies like "llama3"/"linear"/"yarn" are
    # registered), so this must fall through to the plain-theta formula, not the ROPE_INIT_FUNCTIONS
    # lookup. Also: rope_theta lives at cfg.rope_theta for Llama but nested inside cfg.rope_scaling
    # for Qwen3 (Qwen3Config has no top-level rope_theta attribute at all) -- verified directly via
    # AutoConfig.from_pretrained for both model families.
    if rope_type not in (None, "default"):
        rope_fn = ROPE_INIT_FUNCTIONS[rope_type]
        inv_freq, _ = rope_fn(cfg, "cpu")
    else:
        rope_theta = getattr(cfg, "rope_theta", None)
        if rope_theta is None:
            rope_theta = rope_scaling["rope_theta"]
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))

    bundle = {
        "codec": "nsn",
        "n_bits": 1,
        "head_dim": int(head_dim),
        "window_size": int(args.window_size),
        "codebook": codebook.to(torch.float16).contiguous(),  # [256, 8], 4-bit-RTN-degraded, fp16 (matches reference's model.half())
        "inv_freq": inv_freq.to(torch.float32).contiguous(),  # [head_dim/2]
        "model_name": args.model_name,
        # Precision the scale-adjusted norm2 is STORED at, declared here rather than left implicit
        # in the serving code: with a bf16 cache the fp16 codebook promotes that product to fp32,
        # which cost 0.125 bit/channel of metadata carrying no extra information about the data
        #. load_bundle checks this against nsn_quant.NORM2_STORE_DTYPE, so a bundle and a
        # server that disagree fail loudly instead of quietly reconstructing different values.
        "norm2_store_dtype": "float16",
    }
    torch.save(bundle, args.out)
    print(f"wrote {args.out}: codebook={bundle['codebook'].shape} inv_freq={bundle['inv_freq'].shape} "
          f"head_dim={head_dim} window_size={args.window_size}")


if __name__ == "__main__":
    main()
