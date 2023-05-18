import torch


def lanczos_alg(order, loss, k_largest, l_smallest=0, return_precision='32', device=None):
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

    if device is None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    def hvp(params, vec, batch) -> torch.Tensor:
        """
        Computes the Hessian-vector product for a mini-batch from the dataset.
        Should not use functorch as it does not support batch norm: https://pytorch.org/functorch/stable/batch_norm.html
        """
        # Compute original gradient, tracking computation graph
        loss_val = loss(params, batch)
        grad_dict = torch.autograd.grad(loss_val, params, create_graph=True)
        grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict]).to(device)

        # Take the second gradient and mult with vec, Hv
        hessian_vec_prod_dict = torch.autograd.grad(grad_vec, params, grad_outputs=vec, only_inputs=True)
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])
        return hessian_vec_prod

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
        v_ = v if return_precision == '64' else v.type(torch.float32)
        w = hvp(params, v_, batch)
        w = w.to(device=device, dtype=torch.float64)

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
        params_flatten = torch.nn.utils.parameters_to_vector(params)
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

        del vecs, tridiag, eigs_tridiag, eigvecs_triag, params_flatten, init_vec
        return k_largest_eigenvals, k_largest_eigenvecs, l_smallest_eigenvals, l_smallest_eigenvecs

    return lanczos_alg_jitted
