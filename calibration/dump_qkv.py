#!/usr/bin/env python3
"""Standalone HF-transformers Q/K/V capture for rotation calibration.

Prefills each prompt once and dumps post-RoPE Q, post-RoPE K (from the HF
prefill cache) and raw V per layer, in the layout compute_kv_rotation.py
consumes:

  <out>/layer_<id>/q/<chunk_id>.pt   [T, n_heads,  head_dim]  fp16
  <out>/layer_<id>/k/<chunk_id>.pt   [T, kv_heads, head_dim]  fp16
  <out>/layer_<id>/v/<chunk_id>.pt   [T, kv_heads, head_dim]  fp16

One chunk = one prompt, numbered 1..N. Chunk 0 is a dummy 6-token slice of
the first prompt: compute_kv_rotation's "all" mode skips chunk 0, so real
prompts must start at 1.

MXFP4 checkpoints (openai/gpt-oss-20b) are dequantized to bf16 at load via
Mxfp4Config(dequantize=True) — the packed MXFP4 path cannot be captured and
calibration wants high-precision statistics anyway. gpt-oss additionally
runs eager attention (no sdpa support — sinks), chunked prefill (MoE expert
activations OOM on long single-shot prefills), and dumps ONLY the
full-attention layers in local order 0..N_full-1 with a layer_map.json
recording local -> global (sliding-window layers' cache is window-truncated
and stays uncompressed in serving).

Requires only stdlib + torch + transformers (+ pandas for CSV prompts).
"""
from __future__ import annotations

import argparse
import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import torch

GPQA_TMPL = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD. Think step by step before answering.\n\n{Question}\n\n"
    "A) {A}\nB) {B}\nC) {C}\nD) {D}"
)


@contextmanager
def capture_rope_q(model: torch.nn.Module):
    """Patch apply_rotary_pos_emb in the model's modeling module to record
    post-RoPE Q per layer call. K and V come from the prefill cache instead:
    the cache stores post-RoPE K and untouched V."""
    module = importlib.import_module(model.__class__.__module__)
    if not hasattr(module, "apply_rotary_pos_emb"):
        raise AttributeError(
            f"Model module '{module.__name__}' does not expose 'apply_rotary_pos_emb'."
        )
    original = module.apply_rotary_pos_emb
    q_chunks: list[torch.Tensor] = []

    def patched(q, k, *args, **kwargs):
        q_out, k_out = original(q, k, *args, **kwargs)
        q_chunks.append(q_out.detach().to("cpu", dtype=torch.float16))
        return q_out, k_out

    module.apply_rotary_pos_emb = patched
    try:
        yield q_chunks
    finally:
        module.apply_rotary_pos_emb = original


def assemble_q_per_layer(chunks: list[torch.Tensor], n_layers: int) -> torch.Tensor:
    if len(chunks) % n_layers != 0:
        raise RuntimeError(
            f"RoPE hook fired {len(chunks)} times, not a multiple of n_layers={n_layers}"
        )
    per_layer = [torch.cat(chunks[layer::n_layers], dim=2) for layer in range(n_layers)]
    return torch.stack(per_layer, dim=0).squeeze(1)  # [L, Hq, T, d]


@torch.inference_mode()
def run_prefill_qkv_capture(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    full_ids: list[int],
    prefill_chunk: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    seq_len = int(input_ids.shape[-1])
    n_layers = int(model.config.num_hidden_layers)

    with capture_rope_q(model) as q_chunks:
        if prefill_chunk is None:
            cache = model(input_ids=input_ids, use_cache=True).past_key_values
        else:
            cache = None
            for s0 in range(0, seq_len, prefill_chunk):
                out = model(
                    input_ids=input_ids[:, s0 : s0 + prefill_chunk],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values

    q_all = assemble_q_per_layer(q_chunks, n_layers)
    ks, vs = [], []
    for layer in full_ids:
        k = cache.layers[layer].keys.detach().to("cpu", dtype=torch.float16).squeeze(0)
        v = cache.layers[layer].values.detach().to("cpu", dtype=torch.float16).squeeze(0)
        if k.shape[1] != seq_len:
            raise RuntimeError(
                f"layer {layer}: cache T={k.shape[1]} != {seq_len} (not a full-attention layer?)"
            )
        ks.append(k)
        vs.append(v)
    return q_all[full_ids], torch.stack(ks), torch.stack(vs)  # each [L, H, T, d] fp16


def checkpoint_is_mxfp4(config) -> bool:
    qc = getattr(config, "quantization_config", None)
    if qc is None:
        return False
    fmt = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
    return str(fmt).lower().endswith("mxfp4")


def load_prompts(path: Path, num_prompts: int) -> list[str]:
    if path.suffix == ".jsonl":
        prompts = [json.loads(line)["prompt"] for line in path.open() if line.strip()]
    else:
        import pandas as pd

        df = pd.read_csv(path)
        if "Question" in df.columns and "Correct Answer" in df.columns:
            prompts = [
                GPQA_TMPL.format(
                    Question=r["Question"],
                    A=r["Correct Answer"],
                    B=r["Incorrect Answer 1"],
                    C=r["Incorrect Answer 2"],
                    D=r["Incorrect Answer 3"],
                )
                for _, r in df.iterrows()
            ]
        else:
            col = next((c for c in ("text", "prompt") if c in df.columns), None)
            if col is None:
                raise ValueError(f"{path}: no GPQA columns and no 'text'/'prompt' column")
            prompts = df[col].astype(str).tolist()
    return prompts[:num_prompts]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--prompts",
        required=True,
        help="Prompt file: .csv (GPQA columns -> multiple-choice template, else a "
        "'text'/'prompt' column) or .jsonl with a 'prompt' field.",
    )
    ap.add_argument("--num-prompts", type=int, default=198)
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Truncate each tokenized prompt to this many tokens (default: no truncation).",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    is_gptoss = getattr(config, "model_type", "") == "gpt_oss"
    quant_cfg = None
    if checkpoint_is_mxfp4(config):
        from transformers import Mxfp4Config

        quant_cfg = Mxfp4Config(dequantize=True)
        print(f"model {args.model}: MXFP4 checkpoint -> dequantizing at load", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map={"": args.device},
        dtype=getattr(torch, args.dtype),
        attn_implementation="eager" if is_gptoss else None,
        trust_remote_code=True,
        quantization_config=quant_cfg,
    )
    model.eval()

    layer_types = getattr(config, "layer_types", None)
    n_layers = int(config.num_hidden_layers)
    if layer_types is not None and any(t != "full_attention" for t in layer_types):
        full_ids = [i for i, t in enumerate(layer_types) if t == "full_attention"]
        prefill_chunk = 4096
    else:
        full_ids = list(range(n_layers))
        prefill_chunk = None

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if len(full_ids) != n_layers:
        json.dump(
            {"local_to_global": full_ids, "model": args.model},
            open(out / "layer_map.json", "w"),
        )
        print(f"{len(full_ids)}/{n_layers} full-attention layers -> layer_map.json", flush=True)

    prompts = load_prompts(Path(args.prompts), args.num_prompts)
    for pi, p in enumerate(prompts):
        if args.no_chat_template:
            ids = tok(p, return_tensors="pt", add_special_tokens=True).input_ids
        else:
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
            )
            ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        if args.max_seq_len is not None:
            ids = ids[:, : args.max_seq_len]
        q, k, v = run_prefill_qkv_capture(model, ids, full_ids, prefill_chunk)
        chunk = pi + 1
        for li in range(len(full_ids)):
            for name, t in (("q", q[li]), ("k", k[li]), ("v", v[li])):
                d = out / f"layer_{li}" / name
                d.mkdir(parents=True, exist_ok=True)
                torch.save(t.permute(1, 0, 2).contiguous(), d / f"{chunk}.pt")  # [T, H, d]
        if pi == 0:
            # Dummy 6-token chunk 0 so compute's "all" mode (which skips
            # chunk 0) keeps every real prompt.
            for li in range(len(full_ids)):
                for name, t in (("q", q[li]), ("k", k[li]), ("v", v[li])):
                    torch.save(
                        t.permute(1, 0, 2)[:6].contiguous(), out / f"layer_{li}" / name / "0.pt"
                    )
        del q, k, v
        if (pi + 1) % 25 == 0:
            print(f"  captured {pi + 1}/{len(prompts)} (T={ids.shape[1]})", flush=True)
    print(f"DONE dump -> {out}  ({len(prompts)} prompts, {len(full_ids)} layers)", flush=True)


if __name__ == "__main__":
    main()
