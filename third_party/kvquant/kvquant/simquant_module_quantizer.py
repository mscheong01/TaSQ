import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.cluster import KMeans

import torch
import tqdm
from torch.distributions import Normal
from .gpu_kmeans import weighted_kmeans_batch


def round_to_nearest_pole_sim(w, poles):
    """
    w: weight/act values (1d vector)
    poles: tuple of values

    Round the numbers in w to the nearest value in poles.
    """
    stack = []
    for c in poles:
        diff = (w - c).abs()
        stack.append(diff)
    diff = torch.stack(stack)
    idx = diff.argmin(axis=0)
    aug = 0
    freq = []
    for i, c in enumerate(poles):
        aug += (idx == i) * c

    return aug

def get_outliers(
    w,
    channel=-1,
    outlier_threshold_upper=-1,
    outlier_threshold_lower=-1,
    cap_outliers=-1,
    first_few_fp16=-1
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    outlier_threshold_upper: upper outlier thresholds
    outlier_threshold_lower: lower outlier thresholds
    first_few_fp16: number of initial tokens to keep in fp16

    Detect outliers above upper threshold / below lower threshold
    """
    # only use either per-channel or per-token outlier
    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)

    under_lower = w < outlier_threshold_lower
    above_upper = w > outlier_threshold_upper

    outlier_mask = torch.logical_or(under_lower, above_upper)

    if cap_outliers > -1:
        outlier_mask_tmp = outlier_mask.clone()

        zero_point = (outlier_threshold_upper + outlier_threshold_lower) / 2
        distance = (outlier_threshold_upper - outlier_threshold_lower) / 2
        outliers = w * outlier_mask

        values = torch.zeros_like(outliers)
        values[outlier_mask] = ((w - zero_point) / distance)[outlier_mask]

        upper_values, upper_indices = torch.topk(values, 21, dim=-1)
        lower_values, lower_indices = torch.topk(values, 21, dim=-1, largest=False)
        indices_combined = torch.cat((upper_indices, lower_indices), dim=-1)
        values_combined = torch.cat((upper_values, lower_values), dim=-1)

        values2 = torch.zeros_like(outliers)
        values2.scatter_(-1, indices_combined, values_combined)
        outlier_mask = values2 != 0

    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16,:] = True

    return outlier_mask

def get_outliers_dynamic(
    w,
    channel=-1,
    thresh=0.999,
    first_few_fp16=-1
):
    """
    w: weight/act values (1d vector)
    channel: which dimension to share scaling factors along
    thresh: percentile for outlier threshold computation
    first_few_fp16: number of initial tokens to keep in fp16

    Detect outliers above upper threshold / below lower threshold
    """

    t = 1-((1-thresh)/2)
    w = w.float()

    # only use either per-channel or per-token outlier
    outlier_threshold_upper = torch.quantile(w, t, dim=channel)
    outlier_threshold_lower = torch.quantile(w, 1-t, dim=channel)

    outlier_threshold_upper = outlier_threshold_upper.unsqueeze(channel)
    outlier_threshold_lower = outlier_threshold_lower.unsqueeze(channel)

    under_lower = w <= outlier_threshold_lower
    above_upper = w >= outlier_threshold_upper

    outlier_mask = torch.logical_or(under_lower, above_upper)

    if first_few_fp16 > -1:
        outlier_mask[:first_few_fp16,:] = True

    return outlier_mask

# integer quantization function
def quant_fn_zp(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    clamp=False
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    clamp: whether to round and clamp the zeropoint

    Performs simulated integer quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using median to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    rangeval = (maxval - minval)
    qx = (2**bits - 1) / rangeval

    # set offset
    if clamp:
        offset = torch.round(minval * qx)
        offset = offset.clamp(-(2**bits - 1), 0)
    else: # improves accuracy with per-channel key quantization
        offset = minval * qx

    offset = offset.unsqueeze(qchannel)
    qx = qx.unsqueeze(qchannel)

    # need to handle outlier removal
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    # scale and subtract offset
    qinp = torch.round(qx * inp - offset)

    #clipping (just for debugging purposes)
    qinp = torch.clip(qinp, min=0, max=2**bits - 1)

    #rescale
    qinp_out = (qinp + offset) / qx

    # add outliers back
    if include_sparse:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0)
    return qinp_out

def quant_fn_nf(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    nf_lut=None
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    nf_lut: NormalFloat signpost values

    Performs simulated NormalFloat quantization
    """

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using mean to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2
    offset = offset.unsqueeze(qchannel)
    rangeval = rangeval.unsqueeze(qchannel)

    # subtract offset
    inp = inp - offset

    # need to handle outlier removal here due to issues with zeroing out non-outliers
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    #dividing by range to normalize to [-1,1]
    inp_scaled = inp / rangeval

    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), nf_lut)
    qinp_out = Q.reshape(inp.shape).half().cuda()
    qinp_out = qinp_out * rangeval

    # add outliers back
    if include_sparse:
        qinp_out = qinp_out + outliers

    #shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0) #TODO: debug (shouldn't be necessary)

    return qinp_out

def quant_fn_nuq_recon(
    inp,
    bits=8,
    qchannel = -1,
    dynamicquantization=False,
    include_sparse=False,
    outlier_mask=None,
    maxval=-1,
    minval=-1,
    lut=None,
    norm=False,
    normscale=None,
    normoffset=None,
    first_few_fp16=-1
):
    """
    inp: weight/act values (2d matrix)
    bits: number of bits for quantization
    qchannel: which dimension to share scaling factors along
    dynamicquantization: whether to compute scaling factors / outlier thresholds online
    include_sparse: whether to use dense-and-sparse quantization
    outlier_mask: positions of outlier values
    maxval: upper outlier thresholds (if not dynamically computed)
    minval: lower outlier thresholds (if not dynamically computed)
    lut: NUQ signpost values
    norm: whether to use Q-Norm
    normscale: scaling for Q-Norm
    normoffset: shift for Q-Norm
    first_few_fp16: number of initial tokens to keep in fp16

    Performs simulated NUQ quantization
    """

    if first_few_fp16 > -1:
        orig = inp

    # set quantization threshold dynamically
    if dynamicquantization:
        if include_sparse:
            outliers = inp * outlier_mask
            median = torch.median(inp, dim=qchannel).values
            median = median.unsqueeze(qchannel)
            median_mask = median * outlier_mask

            # recenter using mean to avoid having outliers skew quant distribution
            tmp_inp = inp - outliers + median_mask
            maxval = torch.max(tmp_inp, dim=qchannel).values
            minval = torch.min(tmp_inp, dim=qchannel).values
        else:
            maxval = torch.max(inp, dim=qchannel).values
            minval = torch.min(inp, dim=qchannel).values

    # compute offset here:
    offset = (maxval + minval) / 2
    rangeval = (maxval - minval) / 2
    offset = offset.unsqueeze(qchannel)
    rangeval = rangeval.unsqueeze(qchannel)

    # subtract offset
    inp = inp - offset

    # need to handle outlier removal here due to issues with zeroing out non-outliers
    if include_sparse:
        outliers = inp * outlier_mask
        inp = inp - outliers

    #dividing by range to normalize to [-1,1]
    inp_scaled = inp / rangeval

    # round to nearest LUT entry
    lut_cuda = torch.tensor(lut[0]).to(inp_scaled.device)
    Q = round_to_nearest_pole_sim(inp_scaled.flatten(), lut_cuda)
    qinp_out = Q.reshape(inp.shape).float().to(inp_scaled.device)

    if norm:
        normscale = normscale.to(inp_scaled.device)
        normoffset = normoffset.to(inp_scaled.device)
        qinp_out = qinp_out*normscale + normoffset

    # un-normalize
    qinp_out = qinp_out * rangeval

    # add outliers back
    if include_sparse:
        qinp_out[outlier_mask] = 0
        qinp_out = qinp_out + outliers

    #shift by offset
    qinp_out = qinp_out + offset
    qinp_out = torch.nan_to_num(qinp_out, nan=0.0, posinf=0.0, neginf=0.0) #TODO: debug (shouldn't be necessary)

    # leave first few in fp16
    # leave this here for now -> avoids any small perturbations from rescaling
    if first_few_fp16 > -1:
        qinp_out[:first_few_fp16,:] = orig[:first_few_fp16,:]

    return qinp_out.float()

# simquant quantizer (calibration)
class SimQuant:
    def __init__(
                    self,
                    layer,
                    num_coupled,
                    bits,
                    perm=None,
                    tail_w=None,
                    chan_w=None,
                    per_token_norm=False,
                    post_norm_bits=0,
                    head_group=1,
                    ptnorm_weight=False,
                    head_order=None,
                    vocab_probe_weight=1.0,
                    maha_fisher=None,
                    fisher_kmeans_init=False,
                    fisher_native_coord=False,
                    fisher_diag_metric=False,
                    kmeans_iters=100,
                ):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.out = None
        self.nsamples = 0
        self.n_vocab_probe = 0   # rows appended via add_vocab_probe --
                                 # these have no Fisher importance (no real loss/backprop exists
                                 # for a synthetic probe), so quantize() must pad `weight` for them
        # Multiplier on the padded weight value, not a row-repeat count. This preserves vocabulary
        # coverage while controlling each probe row's influence on its centroid.
        self.vocab_probe_weight = vocab_probe_weight

        self.c = num_coupled
        self.b = bits
        # per-head channel permutation, shape (n_kv_heads, head_dim); None = contiguous
        self.perm = perm
        # tail_w: (n_kv_heads, n_tokens) multiplicative k-means weight boost for
        # high-attention key tokens (CVaR-style top-alpha boost)
        self.tail_w = tail_w
        self.chan_w = chan_w                 # (n_kv_heads, head_dim) sqrt(Cq_ii) scale  [W]
        self.per_token_norm = per_token_norm  # normalise each token before k-means      [N]
        self.post_norm_bits = post_norm_bits
        self.head_group, self.head_order = head_group, head_order
        # k-means clusters PER-TOKEN NORMALISED rows z = x/s but weights them by raw-space Fisher
        # importance. The distortion that matters lives in raw space,
        #     delta_x_n = s_n * (z_n - c)   =>   D = sum_n imp_n * s_n^2 * ||z_n - c||^2
        # so the weight has to carry s_n^2. Off by default: it changes every calibrated codebook.
        self.ptnorm_weight = ptnorm_weight
        # Fisher-weighted k-means++ init.
        # Opt-in: it changes the fitted codebook, so it is a separate arm.
        self.fisher_kmeans_init = fisher_kmeans_init
        # divide the Fisher weight by w_i so the k-means objective is a native-space quantity
        # (see the block in quantize()). Opt-in: it changes the fitted codebook.
        self.fisher_native_coord = fisher_native_coord
        # per-channel diagonal metric diag(b_i/w_i) instead of folding the channel
        # dependence into the scalar sample weight
        self.fisher_diag_metric = fisher_diag_metric
        # Lloyd iteration cap.
        self.kmeans_iters = kmeans_iters
        # Full empirical Fisher F, (n_kv_heads, head_dim, head_dim), raw and unpermuted. The metric
        # becomes Mahalanobis (x-c)^T F_G (x-c) per channel-group G instead of Euclidean ||x-c||^2.
        # Permuted/sliced into per-group c x c submatrices in quantize(), same place perm/chan_w
        # are applied, so it stays aligned with whatever grouping make_tasq.py's matching_full chose.
        self.maha_fisher = maha_fisher
        if perm is not None:
            import numpy as _np, torch as _t
            H, Dh = perm.shape
            self._perm_idx = _t.from_numpy(
                (_np.arange(H)[:, None] * Dh + perm).reshape(-1)).long()

    def add_batch(self, inp, out):
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        tmp = out.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(out.shape) == 3:
                out = out.reshape((-1, self.rows))
        self.nsamples += tmp

        if self.out == None:
            self.out = out.clone()
        else:
            self.out = torch.cat((self.out, out.clone()), dim=0)

    def add_vocab_probe(self, out):
        """append synthetic full-vocabulary "[BOS, token]" K/V rows, same mechanics as add_batch, but tracked
        separately since these rows have no Fisher importance -- quantize() pads `weight`
        with the real corpus's mean weight for exactly this many trailing rows."""
        if len(out.shape) == 2:
            out = out.unsqueeze(0)
        if isinstance(self.layer, nn.Linear) and len(out.shape) == 3:
            out = out.reshape((-1, self.rows))
        self.n_vocab_probe += out.shape[0]
        if self.out is None:
            self.out = out.clone()
        else:
            self.out = torch.cat((self.out, out.clone()), dim=0)

    def quantize(
        self,
        include_sparse=False,
        sparsity_threshold=0.999,
        nuq=False,
        fisher=False,
        norm=False,
        cap_outliers=False,
        first_few_fp16=-1
    ):
        torch.cuda.empty_cache()
        self._ptscale = None      # set only by the normalisation block below; never stale
        data = self.out.float()
        fisher = fisher.reshape(-1, fisher.shape[-1])
        if self.per_token_norm or self.chan_w is not None:
            import math as _m, torch as _t, numpy as _np
            H = (self.chan_w.shape[0] if self.chan_w is not None else 1)
            Dh = data.shape[-1] // H
            v = data.view(-1, H, Dh)
            def _sc(x, bits):
                s_ = x.norm(dim=-1, keepdim=True) / _m.sqrt(Dh)
                if bits >= 16:
                    # 16 means STORED AS FP16, not "unrounded": the serving path keeps one raw
                    # float per (layer, token) (SGLANG_MIXED_KV_SCALE_DTYPE), so this is the only
                    # way the codebook can be fit under the scale the decoder will actually
                    # multiply by. Passing it straight through (the old behaviour) fit the
                    # codebook at fp32 while serving stored something else.
                    return s_.half().float().clamp(min=1e-4)
                sh = s_.shape; g = 64 if s_.shape[0] % 64 == 0 else s_.shape[0]
                lv = (1 << bits) - 1
                w_ = s_.reshape(-1, g).float()
                mn, mx = w_.amin(-1, keepdim=True), w_.amax(-1, keepdim=True)
                st = ((mx - mn) / lv).clamp(min=1e-9)
                return ((((w_ - mn) / st).round().clamp(0, lv) * st + mn)
                        .reshape(sh).clamp(min=1e-4))
            ptscale = _t.ones(v.shape[0], H, dtype=v.dtype, device=v.device)
            # One normalisation only, after W -- the pre-weight scale that used to run here was
            # removed along with its `norm_bits` argument (every shipped build passed 0).
            if self.chan_w is not None:                               # [W]
                v = v * _t.as_tensor(self.chan_w, dtype=v.dtype, device=v.device)[None]
            if self.post_norm_bits:                                   # [N] post-weight
                if self.head_group <= 1:
                    s_post = _sc(v, self.post_norm_bits)
                    ptscale = ptscale * s_post[..., 0]
                    v = v / s_post
                else:
                    ho = (self.head_order if self.head_order is not None else _np.arange(H))
                    for i in range(0, H, self.head_group):
                        cl = list(ho[i:i + self.head_group])
                        sub = v[:, cl]                                    # (N,g,Dh)
                        sc = (sub ** 2).mean(dim=(1, 2), keepdim=True).sqrt()
                        if self.post_norm_bits >= 16:
                            # fp16 storage, same convention as _sc above -- NOT a 16-bit block
                            # RTN, which is what falling through here used to compute.
                            sc = sc.half().float().clamp(min=1e-4)
                        else:
                            sh = sc.shape; g = 64 if sc.shape[0] % 64 == 0 else sc.shape[0]
                            lv = (1 << self.post_norm_bits) - 1
                            w_ = sc.reshape(-1, g).float()
                            mn, mx = w_.amin(-1, keepdim=True), w_.amax(-1, keepdim=True)
                            st = ((mx - mn) / lv).clamp(min=1e-9)
                            sc = ((((w_ - mn) / st).round().clamp(0, lv) * st + mn)
                                  .reshape(sh).clamp(min=1e-4))
                        ptscale[:, cl] = ptscale[:, cl] * sc[:, 0]
                        v[:, cl] = sub / sc
            data = v.reshape(data.shape[0], -1)
            self._ptscale = ptscale
        if self.perm is not None:
            # permute activations AND fisher weights identically so importance stays
            # aligned with its channel
            idx = self._perm_idx.to(data.device)
            data = data[..., idx]
            fisher = fisher[..., idx.to(fisher.device)]
        data = data.reshape(-1, data.shape[-1] // self.c, self.c)
        fisher = fisher.reshape(-1, fisher.shape[-1] // self.c, self.c) # (1, 16384, 4096) -> (16384, 1024, 4)
        data = data.transpose(0, 1).contiguous()
        fisher = fisher.transpose(0, 1).contiguous() # (1024, 16384, 4)
        diag_metric = None
        if self.fisher_diag_metric and self.chan_w is not None:
            # The exact objective is per-channel, not a scalar:
            #     sum_i F_i (dk_i)^2 = s^2 sum_i (F_i / w_i) (du~_i)^2 .
            # Factorising F_{t,i} ~= a_t * b_i splits it into the two things k-means CAN represent
            # exactly -- a per-token sample weight a_t*s^2 and a per-channel METRIC diag(b_i/w_i).
            # Note where that leaves the scalar: a_t is proportional to sum_i F_{t,i}, i.e. the
            # ORIGINAL (fcA) weight. Folding 1/w_i into the scalar instead is a different, weaker
            # approximation. So this branch pairs the fcA scalar with the metric.
            import torch as _t
            _b = fisher.mean(dim=1)                                     # (n_groups, c) channel profile
            _w = _t.as_tensor(self.chan_w, dtype=fisher.dtype, device=fisher.device) ** 2
            _w = _w.reshape(-1)
            if self.perm is not None:
                _w = _w[self._perm_idx.to(_w.device)]
            _w = _w.reshape(-1, self.c).clamp(min=1e-12)
            m = (_b / _w).clamp(min=1e-30)
            # Normalise each group to unit geometric mean. Squared gradients are ~1e-10, and the
            # kmeans kernel compares distances in fp16: without this the transformed data underflows
            # and every point looks identical. A per-group scale is free -- it multiplies every
            # distance in that group equally, so the partition and the recovered centroid are
            # unchanged (the same reasoning the full-matrix maha path uses).
            diag_metric = (m / m.log().mean(-1, keepdim=True).exp()).sqrt()   # (n_groups, c)

        if self.fisher_native_coord and self.chan_w is not None:
            # The Fisher statistic is measured against the NATIVE activation k, but the objective
            # below accumulates error in the transformed coordinates u~ = D(k-mu)/s. Since
            # dk_i = (s/sqrt(w_i)) du~_i, the native-space distortion of a group is
            #     sum_i F_i (dk_i)^2 = s^2 sum_i (F_i / w_i) (du~_i)^2,
            # so the per-channel weights carry 1/w_i. The default path omits it, which is exact
            # only when F factorises within a group as (token profile) x (channel profile) -- a
            # group-constant rescaling then, and k-means is invariant to that. This branch keeps
            # the factor, so the two differ exactly by the token-to-token variation in F's
            # within-group channel profile.
            import torch as _t
            _w = _t.as_tensor(self.chan_w, dtype=fisher.dtype, device=fisher.device) ** 2
            _w = _w.reshape(-1)
            if self.perm is not None:
                _w = _w[self._perm_idx.to(_w.device)]
            _w = _w.reshape(-1, self.c).clamp(min=1e-12)          # (n_groups, c)
            weight = (fisher / _w[:, None, :]).sum(dim=-1)        # (n_groups, n_tokens)
        else:
            weight = fisher.sum(dim=-1)                      # (n_groups, n_tokens)
        if self.ptnorm_weight and getattr(self, "_ptscale", None) is not None:
            import torch as _t
            # group g belongs to head g // (n_groups // H) -- the same mapping tail_w uses below
            ps = self._ptscale.to(weight.device).to(weight.dtype)          # (n_tokens, H)
            g_per_head = weight.shape[0] // ps.shape[1]
            weight = weight * (ps ** 2).T.repeat_interleave(g_per_head, dim=0)[:, :weight.shape[1]]
        if self.tail_w is not None:
            import torch as _t
            tw = _t.as_tensor(self.tail_w, dtype=weight.dtype, device=weight.device)
            g_per_head = weight.shape[0] // tw.shape[0]
            weight = weight * tw.repeat_interleave(g_per_head, dim=0)[:, :weight.shape[1]]
        weight /= weight.max()

        if self.n_vocab_probe > 0:
            # `data` has n_vocab_probe extra trailing rows (see
            # add_vocab_probe) that `fisher`/`weight` never covered -- no real loss/backprop
            # exists for a synthetic probe. Pad with each group's own mean weight, so the probe
            # rows get "typical" per-group importance rather than 0 (ignored) or an unbounded
            # default.
            pad = (weight.mean(dim=-1, keepdim=True) * self.vocab_probe_weight
                   ).expand(-1, self.n_vocab_probe)
            weight = torch.cat([weight, pad], dim=-1)

        maha_sqrt = maha_inv_sqrt = None
        if self.maha_fisher is not None:
            # Transform-then-standard-k-means implements the full Fisher metric.
            # Per channel-group G, k-means's Euclidean E-step on x'=F_G^(1/2)x is exactly the
            # Mahalanobis E-step (x-c)^T F_G (x-c) on x -- and since the M-step is a plain (weighted)
            # mean, which is linear, transforming back with F_G^(-1/2) recovers the EXACT correct
            # weighted mean in the original space (not an approximation). Requires zero changes to
            # the fixed-dimension CUDA kmeans kernel (dist_argmin_half_batched_d{4,8,9,10}).
            Hh, Dh, _ = self.maha_fisher.shape
            Fm = torch.as_tensor(self.maha_fisher, dtype=torch.float32, device=data.device)  # (H,Dh,Dh)
            if self.perm is not None:
                perm_t = torch.as_tensor(self.perm, dtype=torch.long, device=data.device)  # (H,Dh)
                Fm = torch.gather(Fm, 1, perm_t.unsqueeze(-1).expand(-1, -1, Dh))   # permute rows
                Fm = torch.gather(Fm, 2, perm_t.unsqueeze(1).expand(-1, Dh, -1))    # permute cols
            NGh = Dh // self.c
            Fm = Fm.reshape(Hh, NGh, self.c, NGh, self.c)
            M_groups = torch.stack([Fm[:, g, :, g, :] for g in range(NGh)], dim=1)  # (H,NGh,c,c)
            M_groups = M_groups.reshape(Hh * NGh, self.c, self.c)                   # (n_groups,c,c)
            w_eig, V_eig = torch.linalg.eigh(M_groups)                              # M = V diag(w) V^T
            eps = (w_eig.amax(-1, keepdim=True) * 1e-4).clamp(min=1e-30)
            sqrt_w = w_eig.clamp(min=eps).sqrt()
            # normalize each group's sqrt-eigenvalues to unit geometric mean -- only the RELATIVE
            # anisotropy (which directions are stretched vs. compressed) encodes the Mahalanobis
            # metric; an overall per-group scale is mathematically free (uniformly rescaling both
            # sqrt_w and its exact reciprocal leaves nearest-neighbor structure and the recovered
            # centroid unchanged). Raw Fisher eigenvalues are ~1e-10 to 1e-7 (squared-gradient
            # scale), so without this, transformed data shrinks to ~1e-5x and underflows in the
            # kmeans kernel's fp16 distance computation -- every point looks identical, converging
            # in 0 iterations (caught by inspecting eigenvalues after the untransformed build
            # suspiciously converged instantly on every single mini-batch).
            sqrt_w = sqrt_w / sqrt_w.log().mean(-1, keepdim=True).exp()
            maha_sqrt = torch.einsum('nij,nj,nkj->nik', V_eig, sqrt_w, V_eig)
            maha_inv_sqrt = torch.einsum('nij,nj,nkj->nik', V_eig, 1.0 / sqrt_w, V_eig)

        centroids = []
        mini_batch = 32
        for i in tqdm.tqdm(range(data.shape[0] // mini_batch)):
            batch = data[i*mini_batch:(i+1)*mini_batch].contiguous()
            dm = None
            if diag_metric is not None:
                # diagonal case: elementwise, so no eigendecomposition and an exact inverse
                dm = diag_metric[i*mini_batch:(i+1)*mini_batch].to(data.device)
                batch = batch * dm[:, None, :]
            if maha_sqrt is not None:
                Ms = maha_sqrt[i*mini_batch:(i+1)*mini_batch].to(data.device)
                Mi = maha_inv_sqrt[i*mini_batch:(i+1)*mini_batch].to(data.device)
                batch = torch.bmm(batch, Ms)
            centroid, labels = weighted_kmeans_batch(batch, weights=weight[i*mini_batch:(i+1)*mini_batch].to(data.device), k=(1 << self.b), num_iters=self.kmeans_iters,
                                                     fisher_init=self.fisher_kmeans_init)
            if maha_sqrt is not None:
                centroid = torch.bmm(centroid, Mi)
            if dm is not None:
                centroid = centroid / dm[:, None, :]
            centroids.append(centroid)
        centroids = torch.cat(centroids, dim=0).cpu()
        return centroids

    def free(self):
        self.out = None
        self.qout = None
        torch.cuda.empty_cache()

# drop-in layer replacement class
class QuantLinearSim(nn.Module):
    def __init__(
                    self,
                    name,
                    bits,
                    quantizer,
                    infeatures,
                    outfeatures,
                    weight,
                    bias,
                    perchannel=True,
                    include_sparse=False,
                    sparsity_threshold=0.999,
                    dynamicquantization=False,
                    nuq=False,
                    nf_nuq=True,
                    norm=False,
                    first_few_fp16=-1,
                    cap_outliers=-1,
                    clamp=False
                ):

        super().__init__()
        if bits not in [2,3,4,5]:
            raise NotImplementedError("Only 3, 4, 5 bits are supported.")
        self.name = name
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits

        self.weight = weight.T.detach().cpu()
        if bias:
            self.bias = bias.detach().cpu()
        else:
            self.bias = None

        self.perchannel = perchannel
        self.dynamicquantization = dynamicquantization
        self.clamp = clamp

        if perchannel:
            self.qchannel = 0
        else: #per-token quant
            self.qchannel = -1

        self.ochannel = self.qchannel

        self.include_sparse = include_sparse
        self.sparsity_threshold = sparsity_threshold
        self.outlier_threshold_upper = torch.tensor(quantizer[0]).cuda().flatten().half()
        self.outlier_threshold_lower = torch.tensor(quantizer[1]).cuda().flatten().half()

        self.nuq = nuq
        self.nf_nuq = nf_nuq
        if self.nuq and not self.nf_nuq:
            self.lut = quantizer[2]
        else:
            self.lut = None

        if norm:
            self.normscale = quantizer[3]
            self.normoffset = quantizer[4]
            self.norm = True
        else:
            self.norm = False
            self.normscale = None
            self.normoffset = None

        self.cap_outliers = cap_outliers
        self.first_few_fp16 = first_few_fp16

        # for normalfloat support - compute NF signposts
        if self.nf_nuq:
            dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
            # get evenly spaced percentile values

            num_signposts_pos = (2 ** (self.bits - 1)) + 1 # for pos half
            num_signposts_neg = (2 ** (self.bits - 1)) # for neg half

            self.nf_signposts_negative = []
            self.nf_signposts_positive = []

            # from https://arxiv.org/pdf/2306.06965.pdf
            offsets = [0.5*(1/32 + 1/30), 1 - 0.5*(1/32 + 1/30)]
            list1 = [offsets[0]]
            spacing = (0.5 - offsets[0]) / (2 ** (self.bits - 1) - 1)

            add = offsets[0]
            for i in range(num_signposts_neg - 1):
                add += spacing
                list1.append(add)

            list2 = []
            spacing = (offsets[1] - 0.5) / (2 ** (self.bits - 1)) #1 extra space
            add = 0.5
            for i in range(num_signposts_pos - 1):
                list2.append(add)
                add += spacing
            list2.append(offsets[-1])

            # first do negative part [0->0.5]
            for i in range(num_signposts_neg):
                v1 = list1[i]
                val = dist.icdf(torch.tensor([v1])).data.numpy()
                self.nf_signposts_negative.append(torch.tensor(val).item())

            # next do positive part [0.5->1]
            for i in range(num_signposts_pos):
                v1 = list2[i]
                val = dist.icdf(torch.tensor([v1])).data.numpy()
                self.nf_signposts_positive.append(torch.tensor(val).item())

            signpost_neg_min = self.nf_signposts_negative[0]
            signpost_neg_max = self.nf_signposts_negative[-1]
            rangeval = abs(signpost_neg_min)-abs(signpost_neg_max)
            off = abs(signpost_neg_max)
            for s in range(len(self.nf_signposts_negative)):
                self.nf_signposts_negative[s] = (self.nf_signposts_negative[s] + off) / rangeval

            signpost_pos_min = self.nf_signposts_positive[0]
            signpost_pos_max = self.nf_signposts_positive[-1]
            rangeval = abs(signpost_pos_max)-abs(signpost_pos_min)
            off = abs(signpost_pos_min)

            for s in range(len(self.nf_signposts_positive)):
                self.nf_signposts_positive[s] = (self.nf_signposts_positive[s] - off) / rangeval

            del self.nf_signposts_positive[0]

            # delete last negative value and merge
            self.nf_signposts = self.nf_signposts_negative + self.nf_signposts_positive

            assert (len(self.nf_signposts) == (2 ** self.bits))

    #replacement forward pass
    def forward(self, x, other_mat=None):

        out_shape = x.shape[:-1] + (self.outfeatures, )
        x = x.reshape(-1,x.shape[-1])

        # copying weight to / from device during evaluation lets us evaluate
        # a large model with limitted memory usage

        self.weight = self.weight.to(x.device)
        if self.bias is not None:
            self.bias = self.bias.to(x.device)

        x = x.half() # for now cast to fp16 and back (quantization code assumes fp32)
        y = x @ self.weight
        y = y + self.bias if self.bias is not None else y
        y = y.float()

        # if using dense-and-sparse quantization, detect outliers in output tensor
        if self.include_sparse:
            if self.dynamicquantization:
                outlier_mask = get_outliers_dynamic(
                    y,
                    channel=self.ochannel,
                    thresh=self.sparsity_threshold,
                    first_few_fp16=self.first_few_fp16
                )
            else:
                self.outlier_threshold_upper = self.outlier_threshold_upper.to(y.device)
                self.outlier_threshold_lower = self.outlier_threshold_lower.to(y.device)
                outlier_mask = get_outliers(
                    y,
                    channel=self.ochannel,
                    outlier_threshold_upper=self.outlier_threshold_upper,
                    outlier_threshold_lower=self.outlier_threshold_lower,
                    cap_outliers=self.cap_outliers,
                    first_few_fp16=self.first_few_fp16
                )
        else:
            outlier_mask = None

        # quantize output tensor
        if self.nuq:
            if self.nf_nuq:
                y = quant_fn_nf(
                    y,
                    bits=self.bits,
                    qchannel=self.qchannel,
                    maxval=self.outlier_threshold_upper,
                    minval=self.outlier_threshold_lower,
                    include_sparse=self.include_sparse,
                    outlier_mask=outlier_mask,
                    dynamicquantization=self.dynamicquantization,
                    nf_lut=self.nf_signposts
                )
            else:
                y = quant_fn_nuq_recon(
                    y,
                    bits=self.bits,
                    qchannel=self.qchannel,
                    maxval=self.outlier_threshold_upper,
                    minval=self.outlier_threshold_lower,
                    include_sparse=self.include_sparse,
                    outlier_mask=outlier_mask,
                    dynamicquantization=self.dynamicquantization,
                    lut=self.lut,
                    norm=self.norm,
                    normscale=self.normscale,
                    normoffset=self.normoffset,
                    first_few_fp16=self.first_few_fp16
                )

        else:
            # low-bit uniform simulated quant
            y = quant_fn_zp(
                y,
                bits=self.bits,
                qchannel=self.qchannel,
                maxval=self.outlier_threshold_upper,
                minval=self.outlier_threshold_lower,
                include_sparse=self.include_sparse,
                outlier_mask=outlier_mask,
                dynamicquantization=self.dynamicquantization,
                clamp=self.clamp
            )

        self.weight = self.weight.cpu()
        if self.bias is not None:
            self.bias = self.bias.cpu()

        y = y.reshape(out_shape)

        y = y.half()
        return y

# update modules
def make_quant_sim(
                    module,
                    quantizers,
                    bits,
                    name='',
                    perchannel=True,
                    include_sparse=False,
                    sparsity_threshold=0.999,
                    dynamicquantization=False,
                    nuq=False,
                    nf_nuq=True,
                    norm=False,
                    cap_outliers=-1,
                    first_few_fp16=-1,
                    clamp=False
                  ):
    if isinstance(module, QuantLinearSim):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers.keys():
            delattr(module, attr)
            setattr(module, attr, QuantLinearSim(
                                                    name1,
                                                    bits,
                                                    quantizers[name1],
                                                    tmp.in_features,
                                                    tmp.out_features,
                                                    tmp.weight,
                                                    tmp.bias is not None,
                                                    perchannel=perchannel,
                                                    include_sparse=include_sparse,
                                                    sparsity_threshold=sparsity_threshold,
                                                    dynamicquantization=dynamicquantization,
                                                    nuq=nuq,
                                                    nf_nuq=nf_nuq,
                                                    norm=norm,
                                                    cap_outliers=cap_outliers,
                                                    first_few_fp16=first_few_fp16,
                                                    clamp=clamp
                                                ))
        del tmp
    for name1, child in module.named_children():
        make_quant_sim(
                        child,
                        quantizers,
                        bits,
                        name + '.' + name1 if name != '' else name1,
                        perchannel=perchannel,
                        include_sparse=include_sparse,
                        sparsity_threshold=sparsity_threshold,
                        dynamicquantization=dynamicquantization,
                        nuq=nuq,
                        nf_nuq=nf_nuq,
                        norm=norm,
                        cap_outliers=cap_outliers,
                        first_few_fp16=first_few_fp16,
                        clamp=clamp
                      )
