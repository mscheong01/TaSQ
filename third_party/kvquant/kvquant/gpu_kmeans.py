import torch
import torch.nn.functional as F


def get_dist_argmin_half_batched(D):
    # Imported lazily: the vendored kmeans_tools .so is built per-node against a specific torch,
    # so a stale one must not stop this module from being imported (e.g. by a test that supplies
    # its own reference assignment kernel).
    import kmeans_tools
    if D == 4:
        return kmeans_tools.dist_argmin_half_batched_d4
    elif D == 8:
        return kmeans_tools.dist_argmin_half_batched_d8
    elif D == 9:
        return kmeans_tools.dist_argmin_half_batched_d9
    elif D == 10:
        return kmeans_tools.dist_argmin_half_batched_d10
    else:
        raise ValueError(f"Unsupported dimension: {D}")


_INIT_ANNOUNCED = False


def _sample_rows(probs, X):
    """Multinomial draw per batch row, with a uniform fallback for degenerate rows.

    torch.multinomial RAISES on an all-zero row, and a row can legitimately go all-zero here:
    every positive-weight vector already sits exactly on a chosen centroid, all weights are zero,
    or there are fewer distinct vectors than requested centroids. Falling back to uniform for
    just those rows keeps the draw defined without perturbing the well-behaved ones.
    """
    B, N = probs.shape
    total = probs.sum(dim=1, keepdim=True)
    degenerate = ~(total > 0).squeeze(1)
    if degenerate.any():
        probs = torch.where(degenerate.unsqueeze(1), torch.ones_like(probs), probs)
        total = probs.sum(dim=1, keepdim=True)
    idx = torch.multinomial(probs / total, num_samples=1).squeeze(1)
    return X[torch.arange(B, device=X.device), idx], idx


def kmeans_plusplus_batch(X, k, weights=None):
    """
    Batch kmeans++ initialization with incremental distance updates.
    X: Tensor of shape (B, N, D)
    weights: optional (B, N) Fisher sample weights. When given, this is Fisher-weighted
        k-means++: the first centroid is drawn with
        P proportional to w_i and each subsequent one with P proportional to w_i * D_i^2,
        where D_i^2 is the squared distance to the nearest already-chosen centroid.

        w_i * D_i^2 is exactly vector i's current contribution to the weighted objective
        J = sum_i w_i ||x_i - c_{z_i}||^2 that the Lloyd steps below already minimize -- so the
        init and the refinement optimize the same thing. Plain k-means++ (weights=None) samples
        on D_i^2 alone, which can spend codewords covering distant but unimportant vectors.

        No channel handling is needed here: X arrives already permuted and channel-scaled, so
        Euclidean distance in this space IS the s_d-weighted distance in the original one.
    Returns: centroids of shape (B, k, D)
    """
    B, N, D = X.shape
    device = X.device

    centroids = torch.empty((B, k, D), device=device)

    if weights is None:
        # First centroid: randomly chosen for each batch.
        random_idx = torch.randint(0, N, (B,), device=device)
        centroids[:, 0] = X[torch.arange(B, device=device), random_idx]
    else:
        w = weights.to(X.dtype).clamp(min=0)
        centroids[:, 0], _ = _sample_rows(w, X)

    # Compute initial distances from the first centroid.
    min_dists = torch.cdist(X, centroids[:, 0:1]).squeeze(2)  # (B, N)

    for i in range(1, k):
        # Probability proportional to the squared distance, times Fisher importance if given.
        probs = min_dists ** 2
        if weights is not None:
            probs = probs * w
        centroids[:, i], _ = _sample_rows(probs, X)

        # Update the minimum distances using the new centroid.
        new_dists = torch.cdist(X, centroids[:, i:i+1]).squeeze(2)
        min_dists = torch.minimum(min_dists, new_dists)

    return centroids


def weighted_kmeans_batch(X, weights, k, num_iters=10, fisher_init=False):
    """
    Batched weighted k-means clustering.
    X: Tensor of shape (B, N, D)
    weights: Tensor of shape (B, N)
    k: number of clusters
    num_iters: maximum iterations
    fisher_init: use Fisher-weighted k-means++ instead of plain k-means++ initialization.
    Returns: centroids of shape (B, k, D) and labels of shape (B, N)
    """
    B, N, D = X.shape
    device = X.device

    # Announce the initialization rule once per process for experiment provenance.
    global _INIT_ANNOUNCED
    if not _INIT_ANNOUNCED:
        print(f"[kmeans] init = {'FISHER-WEIGHTED kmeans++' if fisher_init else 'plain kmeans++'}",
              flush=True)
        _INIT_ANNOUNCED = True
    centroids = kmeans_plusplus_batch(X, k, weights=weights if fisher_init else None)

    # Select the appropriate half-batched distance function.
    dist_argmin_half_batched = get_dist_argmin_half_batched(D)

    for it in range(num_iters):
        # Cluster assignment step using the custom distance kernel.
        labels = dist_argmin_half_batched(X.half(), centroids.half())  # (B, N)
        # Compute weighted sums and counts for centroids using scatter_add.
        weighted_sum = torch.zeros(B, k, D, device=device)
        weighted_count = torch.zeros(B, k, device=device)
        
        # Scatter the weighted sums: expand labels to (B, N, 1) to match X’s dimensions.
        weighted_sum.scatter_add_(1, labels.unsqueeze(-1).expand(B, N, D).long(), weights.unsqueeze(-1) * X)
        weighted_count.scatter_add_(1, labels.long(), weights)
        
        # Update centroids: compute the weighted average.
        centroids_new = weighted_sum / (weighted_count.unsqueeze(-1) + 1e-8)

        zero_mask = (centroids_new == 0).all(dim=-1)  # shape: (B, k)
        rand_indices = torch.randint(0, N, size=(B, k), device=X.device)  # shape: (B, k)
        reinit_vectors = torch.gather(
            X, 1, rand_indices.unsqueeze(-1).expand(-1, -1, D)
        )  # shape: (B, k, D)
        centroids_new = torch.where(
            zero_mask.unsqueeze(-1),  # shape: (B, k, 1)
            reinit_vectors,
            centroids_new
        )

        # Convergence check: if the mean absolute change is small enough, break early.
        if torch.abs(centroids_new - centroids).mean() < 1e-5:
            print(f"Converged at iteration {it}/{num_iters}")
            return centroids_new, labels

        centroids = centroids_new

    return centroids, labels
