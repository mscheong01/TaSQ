import time

import torch
import torch.nn as nn

from kvquant.modelutils import *
from kvquant.datautils import *
from kvquant.simquant_module_quantizer import *

from kvquant.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)

import pickle
import json

import math
import argparse

def _build_tailw(args):
    """CVaR-style weights: boost the top-alpha fraction of keys by attention mass,
    thresholded PER (layer, kv-head) so every head contributes its own tail.

    --tail-continuous switches to a different (non-binary) scheme: the mass/energy value itself
    is used as the k-means M-step weight, mean-normalized per (layer, kv-head) so the average
    weight stays 1 (same effective sample count as uniform weighting), floored at --tail-min-w so
    no token's weight collapses to ~0. Intended for a follow-up study:
    a continuous sum-of-squared-attention "energy" signal
    matches D_V's additive sum-over-queries structure better than the binary top-alpha/boost
    scheme, which was designed for D_K's max-dominated structure."""
    if not args.tail_mass_file:
        return None
    import numpy as _np
    m = pickle.load(open(args.tail_mass_file, 'rb'))['mass']      # (L, H, N)
    if args.tail_continuous:
        mean = m.mean(axis=-1, keepdims=True)
        w = _np.clip(m / mean, args.tail_min_w, args.tail_max_w).astype(_np.float32)
        print(f"[tailw-continuous] mean-normalized weight (min={args.tail_min_w}, "
              f"max={args.tail_max_w}): "
              f"per-head-mean(w)={_np.median(w.mean(axis=-1)):.3f} "
              f"max(w)={w.max():.1f} min(w)={w.min():.3f}")
        return {li: w[li] for li in range(w.shape[0])}
    thr = _np.quantile(m, 1.0 - args.tail_alpha, axis=-1, keepdims=True)
    w = _np.where(m >= thr, args.tail_boost, 1.0).astype(_np.float32)
    print(f"[tailw] alpha={args.tail_alpha} boost={args.tail_boost} "
          f"-> boosted share of total weight ~= "
          f"{(args.tail_alpha*args.tail_boost)/(args.tail_alpha*args.tail_boost+1-args.tail_alpha):.2f}")
    return {li: w[li] for li in range(w.shape[0])}


def get_model(model, seqlen, maxseqlen):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    # Set RoPE scaling factor

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.half, ignore_mismatched_sizes=True)
    # --- fused-QKV architectures (Phi-4-reasoning-plus and anything else packing qkv_proj) -------
    # The chain below reaches for self_attn.{q,k,v}_proj by name. Phi3ForCausalLM packs them, so a
    # shim installs real slice Linears and reroutes qkv_proj through them (calibration/shims/phi4/phi3_shim.py --
    # gated on the real weights by calibration/shims/phi4/test_shim.py). No-op on llama/qwen3.
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "calibration", "shims", "phi4"))
    from inject import maybe_unfuse as _maybe_unfuse
    _maybe_unfuse(model)


    return model

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')
    model_type = parse_model(model)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    embeddings = get_embedding(model, model_type)
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            # nn.Module.__getattr__ only searches _parameters/_buffers/_modules;
            # forward everything else (e.g. Qwen3's per-layer `attention_type`,
            # read by the model forward before calling the layer) to the wrapped
            # module so the Catcher is attribute-transparent.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("module"), name)
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print("Layer", i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            if model_type == 'opt':
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                )[0]
            else:
                assert model_type == 'llama'
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    norm = get_norm(model, model_type)
    if norm is not None:
        norm = norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())
    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def llama_calibration(model, dataloader, dev, perchannel_match, pertensor_match, num_coupled, bits, perms=None, tailw=None, chanw=None, ptnorm=False, pnbits=0, hgroup=1, horder=None, permsv=None, chanwv=None, ptnormv=False, pnbitsv=0, hgroupv=1, horderv=None, include_sparse=False, sparsity_threshold=0.999, nuq=False, fisher=None, norm=False, cap_outliers=False, first_few_fp16=False, vocab_probe=None, vocab_probe_weight=0, vocab_probe_layers=None, vocab_probe_ids=None, fisherfull=None, fisherfullv=None, tailw_v=False, tailw_k=True, ptnorm_weight=False, fisher_kmeans_init=False, kmeans_iters=100,
                      fisher_native_coord=False, fisher_diag_metric=False):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def __getattr__(self, name):
            # nn.Module.__getattr__ only searches _parameters/_buffers/_modules;
            # forward everything else (e.g. Qwen3's per-layer `attention_type`,
            # read by the model forward before calling the layer) to the wrapped
            # module so the Catcher is attribute-transparent.
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(super().__getattr__("module"), name)
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Quantizing ...')

    quantizers = {}
    for i in range(len(layers)):
        print("Layer", i)
        layer = layers[i].to(dev)
        full = find_layers(layer)

        perchannel_list = []
        pertensor_list = []
        full_list = []

        for f in full:
            for p in perchannel_match:
                if p in f:
                    perchannel_list.append(f)
                    full_list.append(f)
            for p in pertensor_match:
                if p in f:
                    pertensor_list.append(f)
                    full_list.append(f)

        sequential = list(full.keys())

        simquant = {}
        subset = {n: full[n] for n in sequential if n in full_list}
        sequential_subset = list(subset.keys())
        for name in sequential:
            # K uses perms/chanw/ptnorm/pnbits/hgroup/horder; V uses the corresponding *v
            # variants. Tail weighting remains K-only unless --tail-w-v is set.
            # D_V is itself a weighted-sum-over-tokens objective, so the same per-token
            # attention-mass weighting used for K's k-means M-step should target V's actual
            # output error directly.
            is_k, is_v = "k_proj" in name, "v_proj" in name
            _pm = perms[i] if (perms is not None and is_k) else (permsv[i] if (permsv is not None and is_v) else None)
            _tw = tailw[i] if (tailw is not None and ((is_k and tailw_k) or (is_v and tailw_v))) else None
            _cw = chanw[i] if (chanw is not None and is_k) else (chanwv[i] if (chanwv is not None and is_v) else None)
            _pn = (bool(ptnorm) and is_k) or (bool(ptnormv) and is_v)
            _pnbits = pnbits if is_k else pnbitsv
            _hgroup = hgroup if is_k else hgroupv
            _ho = ((horder[i] if horder is not None else None) if is_k else
                   (horderv[i] if horderv is not None else None))
            _mf = (fisherfull[i] if (fisherfull is not None and is_k) else
                   (fisherfullv[i] if (fisherfullv is not None and is_v) else None))
            if name in perchannel_list:
                simquant[name] = SimQuant(
                                        subset[name],
                                        num_coupled,
                                        bits,
                                        perm=_pm,
                                        tail_w=_tw,
                                        chan_w=_cw,
                                        per_token_norm=_pn,
                                        post_norm_bits=_pnbits,
                                        head_group=_hgroup,
                                        ptnorm_weight=ptnorm_weight,
                                        fisher_kmeans_init=fisher_kmeans_init,
                                        fisher_native_coord=fisher_native_coord,
                                        fisher_diag_metric=fisher_diag_metric,
                                        kmeans_iters=kmeans_iters,
                                        head_order=_ho,
                                        vocab_probe_weight=vocab_probe_weight,
                                        maha_fisher=_mf,
                                     )
            elif name in pertensor_list:
                simquant[name] = SimQuant(
                                        subset[name],
                                        num_coupled,
                                        bits,
                                        perm=_pm,
                                        tail_w=_tw,
                                        chan_w=_cw,
                                        per_token_norm=_pn,
                                        post_norm_bits=_pnbits,
                                        head_group=_hgroup,
                                        ptnorm_weight=ptnorm_weight,
                                        fisher_kmeans_init=fisher_kmeans_init,
                                        fisher_native_coord=fisher_native_coord,
                                        fisher_diag_metric=fisher_diag_metric,
                                        kmeans_iters=kmeans_iters,
                                        head_order=_ho,
                                        vocab_probe_weight=vocab_probe_weight,
                                        maha_fisher=_mf,
                                     )
            else:
                continue

        # Qwen3 (and any future arch with QK-RMSNorm) applies q_norm/k_norm to the
        # per-head-reshaped q_proj/k_proj output *before* RoPE/quantization touch it
        # (see src/models/qwen3.py's QuantizedQwen3Attention.forward). Hooking k_proj/
        # q_proj's raw Linear output directly -- the pattern below always used for
        # Llama, which has no such norm -- would calibrate on the wrong distribution
        # for Qwen3: the deployed quantizer never sees pre-norm activations. Apply the
        # same norm here, on the same per-head axis, before handing the batch to
        # SimQuant (which still expects the flat Linear-output shape for its
        # weight.shape[0]-derived row bookkeeping -- reshape back after normalizing).
        qk_norm = hasattr(layer.self_attn, "k_norm") and hasattr(layer.self_attn, "q_norm")
        head_dim = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)

        def add_batch(name):
            norm_mod = None
            if qk_norm:
                if name.endswith("k_proj"):
                    norm_mod = layer.self_attn.k_norm
                elif name.endswith("q_proj"):
                    norm_mod = layer.self_attn.q_norm
            def tmp(_, inp, out):
                o = out.data
                if norm_mod is not None:
                    shp = o.shape
                    o = norm_mod(o.view(*shp[:-1], -1, head_dim)).reshape(shp)
                simquant[name].add_batch(inp[0].data, o)
            return tmp
        handles = []

        for name in sequential_subset:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        

        rotary_emb = model.model.rotary_emb
        position_emb = rotary_emb(inps[0], position_ids)
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_emb
            )[0]

        for h in handles:
            h.remove()

        if vocab_probe is not None:
            # Blend a corpus-independent ``[BOS, token]`` vocabulary probe into codebook fitting.
            vp = vocab_probe.get(i, None)
            layer_ok = (vocab_probe_layers is None) or (i in vocab_probe_layers)
            if vp is not None and vocab_probe_weight > 0 and layer_ok:
                # Add every vocabulary row once; vocab_probe_weight controls its k-means weight.
                for name in sequential_subset:
                    is_k, is_v = "k_proj" in name, "v_proj" in name
                    key = "k" if is_k else ("v" if is_v else None)
                    if key is None or key not in vp:
                        continue
                    vt = vp[key]
                    if vocab_probe_ids is not None:
                        vt = vt[vocab_probe_ids]
                    simquant[name].add_vocab_probe(vt.to(dev))

        for name in subset:

            if fisher is not None:
                key = 'model.layers.%d.%s' % (i, name)
                key = key + '.weight'
                fisher_info = fisher[key].cpu()
            else:
                fisher_info = None

            #cap is always onlyK
            if "k_proj" in name:
                if cap_outliers == -1:
                    cap = False
                else:
                    cap = True
            else:
                cap = False

            quantizers['model.layers.%d.%s' % (i, name)] = simquant[name].quantize(
                include_sparse=include_sparse,
                sparsity_threshold=sparsity_threshold,
                nuq=nuq,
                fisher=fisher_info,
                norm=norm,
                cap_outliers=cap,
                first_few_fp16=first_few_fp16
            )
            simquant[name].free()

        layers[i] = layer.cpu()
        del layer
        del simquant
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='llama model to load'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0,
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=16,
        help='Number of calibration data samples.'
    )

    #args for quantizers
    parser.add_argument(
        '--quantize', action='store_true',
        help='Whether to run calibration to quantize the KV cache.'
    )
    parser.add_argument(
        '--num_coupled', type=int, default=4, choices=[2, 4, 8],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--abits', type=int, default=8, choices=[4,5,6,8,9,10],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--nuq', action='store_true',
        help='Whether to use non-uniform quantization.'
    )
    parser.add_argument(
        '--nf', action='store_true',
        help='Whether to use NormalFloat-based non-uniform quantization.'
    )

    #detailed quantization parameters
    parser.add_argument(
        '--perchannel', type=json.loads, default=["k_proj"],
        help='Tensors to use channel-wise quant.'
    )
    parser.add_argument(
        '--pertoken', type=json.loads, default=["v_proj"],
        help='Tensors to use token-wise quant.'
    )
    parser.add_argument(
        '--include_sparse', action='store_true',
        help='Whether to use dense-and-sparse quantization.'
    )
    parser.add_argument(
        '--sparsity-threshold', type=float, default=1,
        help='Outlier percentile.'
    )
    parser.add_argument(
        '--norm', action='store_true',
        help='Whether to use q-norm.'
    )
    parser.add_argument(
        '--tasq-file', type=str, default=None,
        help='pickle with perms + weights from codebooks/make_tasq.py (enables P and W)'
    )
    parser.add_argument(
        '--per-token-norm', action='store_true', help='enable N (per-token scale)'
    )
    parser.add_argument('--post-norm-bits', type=int, default=0)
    parser.add_argument('--head-group', type=int, default=1)
    parser.add_argument(
        '--per-token-norm-v', action='store_true',
        help='enable P/W/N on Values too. Reads v_perms/v_weights/'
             'v_head_order from the same --tasq-file. Only meaningful with --tasq-file.'
    )
    parser.add_argument('--post-norm-bits-v', type=int, default=0)
    parser.add_argument('--head-group-v', type=int, default=1)
    parser.add_argument(
        '--fisher-diag-metric', action='store_true',
        help='carry the per-channel factor as a k-means METRIC diag(b_i/w_i) with b the \n             per-channel mean Fisher, keeping the plain sum_i F_i sample weight. This is \n             the exact treatment under a rank-1 factorisation of F, where --fisher-native- \n             coord is only a scalar approximation of it.'
    )
    parser.add_argument(
        '--fisher-native-coord', action='store_true',
        help='divide the Fisher sample weight by w_i, making the k-means objective a native-space '
             'Fisher distortion rather than a normalized-space one. The TaSQ paper configuration '
             'enables this through codebooks/build_bundles.sh; the standalone CLI default remains '
             'off for backward compatibility.'
    )
    parser.add_argument(
        '--kmeans-iters', type=int, default=100,
        help='maximum number of Lloyd iterations (default: 100)'
    )
    parser.add_argument(
        '--fisher-kmeans-init', action='store_true',
        help='Fisher-weighted k-means++ initialization: '
             'draw the first centroid with P ~ w_i and each next with P ~ w_i * D_i^2, instead '
             'of plain k-means++ on D_i^2 alone. The Lloyd steps were already Fisher-weighted; '
             'this makes initialization optimize the same objective. Off by default.'
    )
    parser.add_argument(
        '--ptnorm-weight', action='store_true',
        help='multiply the k-means sample weight by the per-token normalisation scale squared. '
             'The trainer clusters z = x/s but weights by raw-space Fisher importance; the '
             'distortion that matters is sum_n imp_n * s_n^2 * ||z_n - c||^2, so s_n^2 belongs '
             'in the weight. Uses the SAME RTN-quantised scale the deployed encoder stores.'
    )
    parser.add_argument(
        '--maha-metric', action='store_true',
        help='use the full Fisher matrix '
             '(fisher_full/v_fisher_full from --tasq-file, requires make_tasq.py '
             '--p-source matching_full) as a Mahalanobis metric for k-means distance, instead of '
             'plain Euclidean. Applied to K always; also to V when --per-token-norm-v is set.'
    )
    parser.add_argument(
        '--tail-mass-file', type=str, default=None,
        help='pickle with mass[l,h,t]; enables CVaR top-attention k-means weighting for keys'
    )
    parser.add_argument(
        '--tail-alpha', type=float, default=0.01, help='top fraction of keys to boost'
    )
    parser.add_argument(
        '--tail-boost', type=float, default=30.0, help='multiplicative boost for that fraction'
    )
    parser.add_argument(
        '--tail-w-v', action='store_true',
        help='also apply --tail-mass-file CVaR weights to the V codebook fit'
    )
    parser.add_argument(
        '--tail-continuous', action='store_true',
        help='use a continuous, mean-normalized k-means weight (mass/mean(mass), floored at '
             '--tail-min-w) instead of the binary top-alpha/boost scheme'
    )
    parser.add_argument(
        '--tail-min-w', type=float, default=0.1,
        help='floor for --tail-continuous weights, so no token collapses to ~0 weight'
    )
    parser.add_argument(
        '--tail-max-w', type=float, default=None,
        help='ceiling for --tail-continuous weights (default: unbounded)'
    )
    parser.add_argument(
        '--no-tail-w-k', dest='tail_w_k', action='store_false', default=True,
        help='leave K codebooks unchanged when applying tail weighting to V'
    )
    parser.add_argument(
        '--perm-file', type=str, default=None,
        help='pickle with {"perms": {layer_idx: (n_kv_heads, head_dim) int array}} for keys'
    )
    parser.add_argument(
        '--quantizer-path', type=str, default=None,
        help='Path to load/store quantizer file'
    )

    # calibration parameters
    parser.add_argument(
        '--fisher', type=str, default=None,
        help='fisher information path'
    )
    parser.add_argument(
        '--seqlen', type=int, default=-1,
        help='Sequence length for calibration / eval.'
    )
    parser.add_argument(
        '--maxseqlen', type=int, default=2048,
        help='Maximum sequence length for the model.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load quantized model.'
    )
    parser.add_argument(
        '--dataset', type=str,
        choices=['wikitext2', 'c4', 'mixed', 'gpqa', 'gpqa_code', 'gpqa_diamond',
                 'gpqa_diamond_windowed', 'code'],
        default='wikitext2',
        help='Calibration dataset. "mixed" combines WikiText-2 and C4; "gpqa_code" combines '
             'GPQA with codeparrot/codeparrot-clean. GPQA variants require Hugging Face access.'
    )
    parser.add_argument(
        '--mix-c4-n', type=int, default=4,
        help='With --dataset mixed: how many of --nsamples calibration samples come from c4 '
             '(the rest from wikitext2).'
    )
    parser.add_argument(
        '--vocab-probe-path', type=str, default=None,
        help='pickle from a vocabulary-probe generator: {layer: {"k"/"v": (V, kv_dim)}} '
             'full-vocabulary "[BOS, token]" K/V backstop, blended into k-means fitting alongside '
             'the real corpus. None = off (default).'
    )
    parser.add_argument(
        '--vocab-probe-weight', type=float, default=0.1,
        help='multiplier on each vocab-probe row\'s k-means weight, relative to the real '
             'corpus\'s own mean weight per group (e.g. 0.1 = each probe row counts as 1/10 of '
             'an average real token -- all vocab rows are always included once for coverage; '
             'this controls influence, not row count). Only used when --vocab-probe-path is set.'
    )
    parser.add_argument(
        '--vocab-probe-layers', type=str, default=None,
        help='comma-separated layer indices for the vocabulary probe (default: all layers)'
    )
    parser.add_argument(
        '--vocab-probe-ids-path', type=str, default=None,
        help='pickle containing a 1D list/tensor of token ids to KEEP from the vocab probe '
             '(e.g. only tokens absent from the real calibration corpus). None (default) = use '
             'the full vocabulary.'
    )

    # arguments for capping outliers and for attention sink
    parser.add_argument(
        '--cap_outliers', type=float, default=-1,
        help='Max % of outliers to retain per token.'
    )
    parser.add_argument(
        '--first_few_fp16', type=int, default=-1,
        help='Leave first few outlier tokens.'
    )
    parser.add_argument(
        '--clamp', action='store_true',
        help='Clamp w/ integer quantization'
    )

    DEV = torch.device('cuda:0')

    args = parser.parse_args()

    #load model
    print('Loading model ...')
    if args.load:
        model = get_model(args.model, args.seqlen, args.maxseqlen)
        model.load_state_dict(torch.load(args.load))
        model.eval()
    else:
        model = get_model(args.model, args.seqlen, args.maxseqlen)
        model.eval()

    if args.seqlen != -1:
        model.seqlen = args.seqlen

    model = model.half()
    print('Done.')

    # TODO: once multi-device evaluation framework is set up, at set_devices call here

    #load dataloaders
    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
        mix_c4_n=args.mix_c4_n,
    )

    if args.quantize:

        # run quantization here

        if args.fisher is not None:
            # support both safetensors and pt filetypes for fisher information
            from os import listdir
            from os.path import isfile, join
            onlyfiles = [join(args.fisher, f) for f in listdir(args.fisher) if (('pytorch_model' in f or 'safetensors' in f) and 'index' not in f)]

            mypath = onlyfiles[0]
            if 'safe' in mypath:
                from safetensors.torch import load_file
                fisher = load_file(mypath, device = 'cpu')
                for i in range(1,len(onlyfiles)):
                    d2 = load_file(onlyfiles[i], device = 'cpu')
                    fisher.update(d2)
            else:
                fisher = torch.load(mypath, map_location='cpu')
                for i in range(1,len(onlyfiles)):
                    d2 = torch.load(onlyfiles[i], map_location='cpu')
                    fisher.update(d2)
        else:
            fisher = None

        # run calibration
        quantizers = llama_calibration(
            model,
            dataloader,
            DEV,
            args.perchannel,
            args.pertoken,
            args.num_coupled,
            args.abits,
            perms=(pickle.load(open(args.tasq_file, 'rb'))['perms'] if args.tasq_file else
                   (pickle.load(open(args.perm_file, 'rb'))['perms'] if args.perm_file else None)),
            tailw=_build_tailw(args),
            tailw_v=args.tail_w_v,
            tailw_k=args.tail_w_k,
            chanw=(pickle.load(open(args.tasq_file, 'rb'))['weights'] if args.tasq_file else None),
            ptnorm=args.per_token_norm,
            pnbits=args.post_norm_bits, hgroup=args.head_group,
            ptnorm_weight=args.ptnorm_weight,
            fisher_kmeans_init=args.fisher_kmeans_init,
            fisher_native_coord=args.fisher_native_coord,
            fisher_diag_metric=args.fisher_diag_metric,
            kmeans_iters=args.kmeans_iters,
            horder=(pickle.load(open(args.tasq_file,'rb')).get('head_order') if args.tasq_file else None),
            # V side -- gated on --per-token-norm-v so old tasq-file
            # reuses that only touch K keep working unchanged even though the file now also
            # carries v_perms/v_weights/v_head_order.
            permsv=(pickle.load(open(args.tasq_file, 'rb')).get('v_perms')
                    if (args.tasq_file and args.per_token_norm_v) else None),
            chanwv=(pickle.load(open(args.tasq_file, 'rb')).get('v_weights')
                    if (args.tasq_file and args.per_token_norm_v) else None),
            ptnormv=args.per_token_norm_v,
            pnbitsv=args.post_norm_bits_v, hgroupv=args.head_group_v,
            horderv=(pickle.load(open(args.tasq_file, 'rb')).get('v_head_order')
                     if (args.tasq_file and args.per_token_norm_v) else None),
            # "final version" Mahalanobis codebook metric -- requires
            # make_tasq.py --p-source matching_full to have populated these keys.
            fisherfull=(pickle.load(open(args.tasq_file, 'rb')).get('fisher_full')
                        if (args.tasq_file and args.maha_metric) else None),
            fisherfullv=(pickle.load(open(args.tasq_file, 'rb')).get('v_fisher_full')
                         if (args.tasq_file and args.maha_metric and args.per_token_norm_v) else None),
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            nuq=args.nuq,
            fisher=fisher,
            norm=args.norm,
            cap_outliers=args.cap_outliers,
            first_few_fp16=args.first_few_fp16,
            vocab_probe=(pickle.load(open(args.vocab_probe_path, 'rb'))
                         if args.vocab_probe_path else None),
            vocab_probe_weight=args.vocab_probe_weight,
            vocab_probe_layers=(set(int(x) for x in args.vocab_probe_layers.split(','))
                                 if args.vocab_probe_layers else None),
            vocab_probe_ids=(pickle.load(open(args.vocab_probe_ids_path, 'rb'))
                              if args.vocab_probe_ids_path else None),
        )

        with open(args.quantizer_path, 'wb') as handle:
            pickle.dump(quantizers, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:

        # load quantizers and evaluate model

        with open(args.quantizer_path, 'rb') as handle:
            quantizers = pickle.load(handle)

        # replace layers
        perchannelquant = {}
        pertokenquant = {}

        perchannel_match = args.perchannel
        pertoken_match = args.pertoken

        for k in quantizers.keys():
            # quantizers[k] = quantizers[k] + (-1, ) # empty for now (used to be LN params)

            # filter out tensor list
            for p in perchannel_match:
                if p in k:
                    perchannelquant[k] = quantizers[k]

            for p in pertoken_match:
                if p in k:
                    pertokenquant[k] = quantizers[k]

        #per-vector quant
        make_quant_sim(
            model,
            perchannelquant,
            args.abits,
            perchannel=True,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=False,
            nuq=args.nuq,
            nf_nuq=args.nf,
            norm=args.norm,
            cap_outliers=args.cap_outliers,
            first_few_fp16=args.first_few_fp16,
            clamp=args.clamp
        )

        #per-vector quant
        make_quant_sim(
            model,
            pertokenquant,
            args.abits,
            perchannel=False,
            include_sparse=args.include_sparse,
            sparsity_threshold=args.sparsity_threshold,
            dynamicquantization=True,
            nuq=args.nuq,
            nf_nuq=args.nf,
            norm=args.norm,
            cap_outliers=args.cap_outliers,
            first_few_fp16=args.first_few_fp16,
            clamp=args.clamp
        )

        #run evaluation
        llama_eval(model, testloader, DEV)
