import functorch
import torch
from torch.autograd import Variable
from torch.utils._pytree import tree_map

from timeit import default_timer as timer
import matplotlib.pyplot as plt

from lanczos_algorithm import lanczos_alg

torch.set_default_dtype(torch.float32)


def lanczos_algorithm_test():

    def get_batch(input_size, output_size, batch_size):
        xs = torch.normal(0., 1., size=(batch_size, input_size))
        ys = torch.randint(output_size, size=(batch_size,))
        ys = torch.eye(output_size)[ys]
        return (xs.to(device), ys.to(device))

    class Net(torch.nn.Module):
        def __init__(self, input_size, output_size, width):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_size, width)
            self.act = torch.nn.Tanh()
            self.fc2 = torch.nn.Linear(width, output_size)
            self.out = torch.nn.LogSoftmax(dim=1)
            self.apply(self._init_weights)

        def forward(self, x):
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            x = self.fc1(x)
            x = self.act(x)
            x = self.fc2(x)
            return self.out(x)

        # Same as JAX initialization
        def _init_weights(self, module):
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_normal_(module.weight)
                torch.nn.init.normal_(module.bias)

    def prepare_single_layer_model(input_size, output_size, width):
        net = Net(input_size, output_size, width).to(device)
        predict, params = functorch.make_functional(net)
        return predict, params

    def full_hessian(loss_fn, params, batch):
        loss = lambda params: loss_fn(params, batch)
        n_params = torch.nn.utils.parameters_to_vector(params).shape[0]
        loss_val = loss(params)
        grad_dict = torch.autograd.grad(loss_val, params, create_graph=True)
        grad_vec = torch.cat([g.view(-1) for g in grad_dict])
        hess_log_target_val = []
        for i in range(n_params):
            deriv_i_wrt_grad = torch.autograd.grad(grad_vec[i], params, retain_graph=True)
            hess_log_target_val.append(torch.cat([h.view(-1) for h in deriv_i_wrt_grad]))
        hessian_matrix = torch.cat(hess_log_target_val, 0).reshape(n_params, n_params)

        return hessian_matrix

    input_size = 10
    output_size = 10
    width = 100
    batch_size = 8
    atol_e = 1e-4

    predict, params = prepare_single_layer_model(input_size, output_size, width)
    num_params = torch.nn.utils.parameters_to_vector(params).shape[0]
    b = get_batch(input_size, output_size, batch_size)

    def loss_fn(params, batch):
        return -torch.mean(predict(params, batch[0]) * batch[1])

    largest_k = 10
    smallest_k = 3
    lanczos_order = 100
    hvp_cl = lanczos_alg(lanczos_order, loss_fn, lanczos_order, return_precision='32')  # Return all lanczos_order eigen products

    # compute the full hessian
    hessian = full_hessian(loss_fn, params, b)
    eigs_true, eigvecs_true = torch.linalg.eigh(hessian)

    for i in range(10):
        start_iteration = timer()
        print("About to run lanczos for", num_params, "parameters")
        eigs_lanczos, eigvecs_lanczos, _, _ = hvp_cl(params, b)

        if i == 0:
            assert torch.allclose(eigs_true[-largest_k:], eigs_lanczos[-largest_k:], atol=atol_e), print("i:", i, "eigs_true:", eigs_true[-largest_k:], "eigs_lanczos:", eigs_lanczos[-largest_k:])
            assert torch.allclose(eigs_true[:smallest_k], eigs_lanczos[:smallest_k], atol=atol_e), print("i:", i, "eigs_true:", eigs_true[:smallest_k], "eigs_lanczos:", eigs_lanczos[:smallest_k])
            perfect_vectors_similarity = torch.eye(largest_k).to(device)
            top_vectors_similarity = eigvecs_lanczos[-largest_k:] @ eigvecs_true[:, -largest_k:]
            assert torch.allclose(perfect_vectors_similarity, torch.abs(top_vectors_similarity), atol=atol_e)
            perfect_vectors_similarity = torch.eye(smallest_k).to(device)
            bottom_vector_similarity = eigvecs_lanczos[:smallest_k] @ eigvecs_true[:, :smallest_k]
            assert torch.allclose(perfect_vectors_similarity, torch.abs(bottom_vector_similarity), atol=atol_e)

        lambda_min, lambda_max = eigs_lanczos[0], eigs_lanczos[-1]
        end_iteration = timer()
        print("iterations", i, ": lambda min:", lambda_min, "lambda max:", lambda_max, "time(s):", end_iteration - start_iteration)

        if i == 0:
            fig, ax = plt.subplots(1, 1)
            eigs_true_edges = torch.concat((eigs_true[:lanczos_order//2], eigs_true[-lanczos_order//2:]))
            ax.plot(range(eigs_lanczos.shape[0]), torch.abs(eigs_true_edges - eigs_lanczos).detach().cpu() / torch.abs(eigs_true_edges).detach().cpu())
            ax.set_title("Accuracy of lanczos eigenvalues")
            ax.set_xlabel("eigenvalue index")
            ax.set_ylabel("| eig_true - eig_lanczos | / eig_true")
            plt.show()

        params = tree_map(lambda p: p * 0.99, params)

    print("True lambda min:", torch.min(eigs_true), "true lambda max:", torch.max(eigs_true))


def lanczos_eigen_approx_test():
    n_dim = 1500
    atol_e = 1e-4
    lanczos_order = 100
    largest_k = 8
    smallest_k = 8

    def fill_diagonal(a, val):
        assert a.ndim >= 2
        k = a.size(0)
        a.as_strided([k], [k + 1]).copy_(val)

    eigenvectors = torch.eye(n_dim)

    eigenvectors[0, 0] = 0.5
    eigenvectors[0, 1] = 0.5
    eigenvectors[1, 0] = -0.5
    eigenvectors[1, 1] = 0.5
    eigenvectors = eigenvectors / torch.linalg.norm(eigenvectors, axis=1)

    eigenvalues = torch.normal(0.0, 1.0, (n_dim,))
    eigenvalues[0] = 10
    eigenvalues[1] = 9
    eigenvalues_matrix = torch.zeros_like(eigenvectors)
    fill_diagonal(eigenvalues_matrix, eigenvalues)
    hessian = (eigenvectors.T @ eigenvalues_matrix @ eigenvectors).to(device)

    eigs_true, eigvecs_true = torch.linalg.eigh(hessian)
    eigs_true, eigvecs_true = eigs_true.to(device), eigvecs_true.to(device)

    def objective(x, batch=None):
        return 0.5 * torch.matmul(torch.matmul(x[0], hessian), x[0])

    x_initial = torch.ones(n_dim) * 0.5
    x_initial[1] = 1.0
    x_initial = x_initial @ eigenvectors
    x_initial = Variable(x_initial.data, requires_grad=True).to(device)

    hvp_cl = lanczos_alg(lanczos_order, objective, lanczos_order, return_precision='32')  # Return all lanczos_order eigen products
    eigs_lanczos, eigvecs_lanczos, _, _ = hvp_cl((x_initial,), None)

    assert torch.allclose(eigs_true[-largest_k:], eigs_lanczos[-largest_k:], atol=atol_e), print("eigs_true:", eigs_true[-largest_k:], "eigs_lanczos:", eigs_lanczos[-largest_k:])
    assert torch.allclose(eigs_true[:smallest_k], eigs_lanczos[:smallest_k], atol=atol_e), print("eigs_true:", eigs_true[:smallest_k], "eigs_lanczos:", eigs_lanczos[:smallest_k])

    perfect_vectors_similarity = torch.eye(largest_k).to(device)
    top_vectors_similarity = eigvecs_lanczos[-largest_k:] @ eigvecs_true[:, -largest_k:]
    assert torch.allclose(perfect_vectors_similarity, torch.abs(top_vectors_similarity), atol=atol_e)

    perfect_vectors_similarity = torch.eye(smallest_k).to(device)
    bottom_vectors_similarity = eigvecs_lanczos[:smallest_k] @ eigvecs_true[:, :smallest_k]
    assert torch.allclose(perfect_vectors_similarity, torch.abs(bottom_vectors_similarity), atol=atol_e)

    fig, ax = plt.subplots(1, 1)
    eigs_true_edges = torch.cat((eigs_true[:lanczos_order // 2], eigs_true[-lanczos_order // 2:]))
    ax.plot(range(eigs_lanczos.shape[0]), torch.abs(eigs_true_edges - eigs_lanczos).detach().cpu() / torch.abs(eigs_true_edges).detach().cpu())
    ax.set_title("Accuracy of lanczos eigenvalues")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("| eig_true - eig_lanczos | / eig_true")
    plt.show()


if __name__ == '__main__':
    device = torch.device("cuda")  # "cpu" or "cuda"
    lanczos_algorithm_test()
    lanczos_eigen_approx_test()
