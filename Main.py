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
        # Start with the original vector
        V[:, i] = U[:, i]

        # Subtract the projection of V[:, i] onto each of the previous vectors V[:, j]
        for j in range(i):
            proj = np.dot(V[:, j], U[:, i]) / la.norm(V[:, j])**2
            V[:, i] -= proj * V[:, j]

        # Normalize the vector if it's not a zero vector
        if la.norm(V[:, i]) > 1e-15:  # small threshold to handle numerical errors
            V[:, i] /= la.norm(V[:, i])

    return V

# Example usage:
# U = np.array([[1, 1], [0, 1]])
# orthogonalized_U = orthogonalize_matrix(U)
# print(orthogonalized_U)
