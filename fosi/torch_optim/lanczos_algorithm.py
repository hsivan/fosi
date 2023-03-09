import torch
from functorch import jvp, grad


def ravel(vec):
    views = []
    for p in vec:
        view = p.view(-1)
        views.append(view)
    return torch.cat(views, 0)


def _zeros_like(ref, device):
    return tuple([torch.zeros_like(p, memory_format=torch.contiguous_format, device=device) for p in ref])


def unravel(vec, ref, device):
    vec_unravel = _zeros_like(ref, device)
    offset = 0
    for p in vec_unravel:
        numel = p.numel()
        # view as to avoid deprecated pointwise semantics
        p.copy_(vec[offset:offset + numel].view_as(p))
        offset += numel
    return vec_unravel


def lanczos_alg(order, loss, k_largest, l_smallest=0, return_precision='32', device=torch.device("cpu")):
    """
    Lanczos algorithm for tridiagonalizing a real symmetric matrix, using full reorthogonalization.
    This function returns a function that performs the Lanczos iterations and can be jitted.
    Args:
        order: an integer corresponding to the number of Lanczos steps to take.
        loss: the loss function used to build the hvp operator (loss function to derive).
        k_largest: an integer corresponding to the required number of the largest eigenvalues and eigenvectors.
        l_smallest: an integer corresponding to the required number of the smallest eigenvalues and eigenvectors.
        return_precision: the algorithm must run in high precision; however, after extracting the eigenvalues and
            eigenvectors we can cast it back to 32/64 bit according to the precision required in return_precision.
    Returns:
        lanczos_alg_jitted: a function that performs the Lanczos algorith and can be jitted.
    """

    # The algorithm must run in high precision; however, after extracting the eigenvalues and eigenvectors we can
    # cast it back to 32 bit.
    # TODO: use torch.float64 as dtype

    def orthogonalization(vecs, w, tridiag, i):
        # Full reorthogonalization.
        # Note that orthogonalization here includes all vectors in vecs, and not just vectors j s.t. j <= i.
        # Since all vectors j s.t. j > i are zeros (vecs is initialized to zeros), there is no impact on w if iteration
        # continues for j > i.
        # However, using the iteration on all the vectors enables us to use jit over this function.
        # Otherwise, we will have to iterate/slice by i, which is not supported by jit.

        # The operation torch.dot(torch.dot(vecs, w), vecs) is equivalent to multiply (scale) each vector in its own coeff,
        # where coeffs = torch.dot(vecs, w) is array of coeffs (scalars) with shape (order,), and then sum all the
        # scaled vectors.
        w = w - vecs.t().matmul(vecs.matmul(w))  # torch.dot(torch.dot(vecs, w), vecs)  # single vector with the shape of w
        # repeat the full orthogonalization for stability
        w = w - vecs.t().matmul(vecs.matmul(w))  # single vector with the shape of w

        beta = torch.linalg.norm(w)

        tridiag[i, i + 1] = beta
        tridiag[i + 1, i] = beta
        vecs[i + 1] = (w / beta).squeeze()

        return (tridiag, vecs)

    def lanczos_iteration(i, args, params, batch):
        vecs, tridiag = args

        # Get last two vectors
        v = vecs[i]

        # Assign to w the Hessian vector product Hv. Uses forward-over-reverse mode for computing Hv.
        # We assume here that the default precision is 32 bit.
        v_not_flat = unravel(v, params, device) if return_precision == '64' else unravel(v.type(torch.float32), params, device)  # convert v to the param tree structure
        loss_fn = lambda x: loss(x, batch)
        hessian_vp = jvp(grad(loss_fn), (params,), (v_not_flat,))[1]  # hvp(loss_fn, params, v_not_flat)[1]
        w = ravel(hessian_vp)
        w = w.type(torch.float64)

        # Evaluates alpha and update tridiag diagonal with alpha
        alpha = torch.dot(w, v)
        tridiag[i, i] = alpha

        w = torch.unsqueeze(w, -1)

        # For every iteration except the last one, perform full orthogonalization on w and normalized it (beta is w's
        # norm). Update tridiag secondary diagonals with beta and update vecs with the normalized orthogonal w.
        if i + 1 < order:
            tridiag, vecs = orthogonalization(vecs, w, tridiag, i)

        return (vecs, tridiag)

    def lanczos_alg_jitted(params, batch):
        """
        Lanczos algorithm for tridiagonalizing a real symmetric matrix, using full reorthogonalization.
        The first time the function is called it is compiled, which can take ~30 second for 10,000,000 parameters
        and order (m) 100.
        Args:
            params: values of the model/function parameters. The gradient of the loss at this params value is used
                in the hvp operator.
            batch: a batch of samples that determines the actual loss function (each loss_i is determined by a
                batch_i of samples, and we use a specific loss_i).
        Returns:
            k_largest_eigenvals: approximate k largest eigenvalues of the Hessian of loss_i at the point params.
            k_largest_eigenvecs: approximate k eigenvectors corresponding to the largest eigenvalues.
            l_smallest_eigenvals: approximate l smallest eigenvalues of the Hessian of loss_i at the point params.
            l_smallest_eigenvecs: approximate l eigenvectors corresponding to the smallest eigenvalues.
        """

        # Initialization
        params_flatten = ravel(params)
        num_params = params_flatten.shape[0]
        tridiag = torch.zeros((order, order), dtype=torch.float64).to(device)
        vecs = torch.zeros((order, num_params), dtype=torch.float64).to(device)
        init_vec = torch.normal(mean=0.0, std=1.0, size=(num_params,), dtype=torch.float64).to(device)
        init_vec = init_vec / torch.linalg.norm(init_vec)
        vecs[0] = init_vec

        lanczos_iter = lambda i, args: lanczos_iteration(i, args, params, batch)
        # Lanczos iterations.
        for i in range(order):
            vecs, tridiag = lanczos_iter(i, (vecs, tridiag))

        eigs_tridiag, eigvecs_triag = torch.linalg.eigh(tridiag)  # eigs_tridiag are also eigenvalues of  the Hessian

        precision = torch.float64 if return_precision == '64' else torch.float32
        k_largest_eigenvals = eigs_tridiag[order-k_largest:].type(precision)
        k_largest_eigenvecs = (eigvecs_triag.T[order-k_largest:] @ vecs).type(precision)
        l_smallest_eigenvals = eigs_tridiag[:l_smallest].type(precision)
        l_smallest_eigenvecs = (eigvecs_triag.T[:l_smallest] @ vecs).type(precision)

        return k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs

    return lanczos_alg_jitted


