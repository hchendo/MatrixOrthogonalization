import numpy as np
import numpy.linalg as la

def orthogonalize(U):
    """
    Orthogonalizes the matrix U using Gram-Schmidt Orthogonalization.

    Args:
        U (numpy.array): A 2D array (matrix) whose columns are to be orthogonalized.

    Returns:
        numpy.array: The orthogonalized matrix.
    """
    # Ensure U is a float type to handle division correctly
    U = U.astype(float)

    # Number of vectors (columns in U)
    num_vectors = U.shape[1]

    # Initialize an empty matrix for the orthogonalized vectors
    V = np.zeros_like(U)

    for i in range(num_vectors):
    V[:, i] = U[:, i]
    for j in range(i):
        norm_Vj = la.norm(V[:, j])
        if norm_Vj > 1e-15:  # small threshold to handle numerical errors
            proj = np.dot(V[:, j], U[:, i]) / norm_Vj**2
            V[:, i] -= proj * V[:, j]

    if la.norm(V[:, i]) > 1e-15:
        V[:, i] /= la.norm(V[:, i])

    return V

# Example usage:
# U = np.array([[1, 1], [0, 1]])
# orthogonalized_U = orthogonalize_matrix(U)
# print(orthogonalized_U)
