import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Workaround to resolve collision between torch and matplotlib
import torch
import matplotlib.pyplot as plt
from lanczos_algorithm import lanczos_alg

torch.set_default_dtype(torch.float32)


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
    hessian = eigenvectors.T @ eigenvalues_matrix @ eigenvectors

    eigs_true, eigvecs_true = torch.linalg.eigh(hessian)

    def objective(x, batch=None):
        return 0.5 * torch.matmul(torch.matmul(x[0], hessian), x[0])

    x_initial = torch.ones(n_dim) * 0.5
    x_initial[1] = 1.0
    x_initial = x_initial @ eigenvectors

    hvp_cl = lanczos_alg(lanczos_order, objective, lanczos_order)  # Return all lanczos_order eigen products
    eigs_lanczos, eigvecs_lanczos, _, _ = hvp_cl((x_initial,), None)

    assert torch.allclose(eigs_true[-largest_k:], eigs_lanczos[-largest_k:], atol=atol_e), print("eigs_true:", eigs_true[-largest_k:], "eigs_lanczos:", eigs_lanczos[-largest_k:])
    assert torch.allclose(eigs_true[:smallest_k], eigs_lanczos[:smallest_k], atol=atol_e), print("eigs_true:", eigs_true[:smallest_k], "eigs_lanczos:", eigs_lanczos[:smallest_k])

    perfect_vectors_similarity = torch.eye(largest_k)
    top_vectors_similarity = eigvecs_lanczos[-largest_k:] @ eigvecs_true[:, -largest_k:]
    assert torch.allclose(perfect_vectors_similarity, torch.abs(top_vectors_similarity), atol=atol_e)

    perfect_vectors_similarity = torch.eye(smallest_k)
    bottom_vectors_similarity = eigvecs_lanczos[:smallest_k] @ eigvecs_true[:, :smallest_k]
    assert torch.allclose(perfect_vectors_similarity, torch.abs(bottom_vectors_similarity), atol=atol_e)

    fig, ax = plt.subplots(1, 1)
    eigs_true_edges = torch.cat((eigs_true[:lanczos_order // 2], eigs_true[-lanczos_order // 2:]))
    ax.plot(range(eigs_lanczos.shape[0]), torch.abs(eigs_true_edges - eigs_lanczos) / torch.abs(eigs_true_edges))
    ax.set_title("Accuracy of lanczos eigenvalues")
    ax.set_xlabel("eigenvalue index")
    ax.set_ylabel("| eig_true - eig_lanczos | / eig_true")
    plt.show()


if __name__ == '__main__':
    lanczos_eigen_approx_test()
