import torch
from torch import nn
import torch.nn.functional as F
import torch.linalg as linalg
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import normalize


def compute_diem(vector_a, vector_b, v_min=0.0, v_max=1.0):
    """
    Compute the Dimension Insensitive Euclidean Metric (DIEM) between two tensors.

    Parameters:
    - vector_a (torch.Tensor): First input tensor, shape (N, D) or (D,)
    - vector_b (torch.Tensor): Second input tensor, shape (N, D) or (D,)
    - v_min (float): Minimum possible value of vector elements.
    - v_max (float): Maximum possible value of vector elements.

    Returns:
    - diem_value (torch.Tensor): DIEM values for each row (if 2D) or a single value (if 1D).
    """

    # Ensure the input tensors have the same shape
    assert vector_a.shape == vector_b.shape, "Input tensors must have the same shape"

    # If inputs are 1D, add an extra batch dimension
    if vector_a.dim() == 1:
        vector_a = vector_a.unsqueeze(0)  # Shape becomes (1, D)
        vector_b = vector_b.unsqueeze(0)  # Shape becomes (1, D)

    # Dimensionality of the vectors (D) and batch size (N)
    N, D = vector_a.shape

    # Compute the Euclidean distance between corresponding rows
    euclidean_dist = torch.norm(vector_a - vector_b, p=2, dim=1)  # Shape: (N,)

    # Expected Euclidean distance for random vectors in D-dimensional space
    expected_dist = ((v_max - v_min) * D) ** 0.5

    # Variance of Euclidean distance
    variance_dist = (D * (v_max - v_min) ** 2) / 12

    # Compute DIEM values for each row
    diem_values = ((v_max - v_min) / variance_dist) * (euclidean_dist - expected_dist)

    return diem_values
    
    
    
def mean_std_entropy(diem_values):   
    # Compute mean, standard deviation, and entropy
    mean_diem = diem_values.mean().item()
    std_diem = diem_values.std().item()

    # Normalize DIEM values to get probability distribution for entropy calculation
    diem_probs = F.softmax(diem_values, dim=0)  # Convert to probability distribution
    entropy_diem = (-diem_probs * torch.log(diem_probs + 1e-9)).sum().item()  # Compute entropy

    return mean_diem, std_diem, entropy_diem


def median_absolute_deviation(diem_values):
    """
    Calculate the Median Absolute Deviation (MAD) of DIEM values.

    Parameters:
    - diem_values (torch.Tensor): Tensor of DIEM values.

    Returns:
    - mad (float): Median Absolute Deviation of the DIEM values.
    """
    median = torch.median(diem_values)
    abs_deviation = torch.abs(diem_values - median)
    mad = torch.median(abs_deviation).item()
    return mad

def soft_cosine_similarity(matrix1, matrix2, feature_similarity):
    """
    Compute the Soft Cosine Similarity between two matrices.

    Parameters:
    - matrix1 (torch.Tensor): First input matrix of shape (N, D).
    - matrix2 (torch.Tensor): Second input matrix of shape (M, D).
    - feature_similarity (torch.Tensor): Feature similarity matrix of shape (D, D).

    Returns:
    - soft_cosine_sim (torch.Tensor): Soft cosine similarity matrix of shape (N, M).
    """
    # Ensure the feature similarity matrix is symmetric
    assert torch.allclose(feature_similarity, feature_similarity.T), "Feature similarity matrix must be symmetric"

    # Compute the adjusted dot product
    adjusted_dot_product = torch.matmul(matrix1, torch.matmul(feature_similarity, matrix2.T))

    # Compute the norms
    norm1 = torch.sqrt(torch.sum(matrix1 * torch.matmul(matrix1, feature_similarity), dim=1, keepdim=True))
    norm2 = torch.sqrt(torch.sum(matrix2 * torch.matmul(matrix2, feature_similarity), dim=1, keepdim=True))

    # Compute the outer product of norms
    norm_product = torch.matmul(norm1, norm2.T)

    # Compute the soft cosine similarity
    soft_cosine_sim = adjusted_dot_product / (norm_product + 1e-8)  # Add epsilon to avoid division by zero

    return soft_cosine_sim

def spectral_metrics(similarity_matrix):
    """
    Compute eigenvalue-based spectral metrics for correlation analysis.

    Parameters:
    - similarity_matrix (torch.Tensor): Soft cosine similarity matrix of shape (N, N).

    Returns:
    - largest_eigenvalue (float): Largest eigenvalue of the normalized Laplacian.
    - trace_laplacian (float): Sum of all eigenvalues (measuring total variance).
    - frobenius_norm (float): Magnitude of overall similarity.
    """
    degree_matrix = torch.diag(torch.sum(similarity_matrix, dim=1))
    laplacian_matrix = degree_matrix - similarity_matrix

    # Compute eigenvalues
    eigenvalues = linalg.eigvalsh(laplacian_matrix)

    # Compute metrics
    largest_eigenvalue = torch.max(eigenvalues).item()
    trace_laplacian = torch.sum(eigenvalues).item()  # Sum of eigenvalues
    frobenius_norm = torch.norm(similarity_matrix, p="fro").item()  # Overall similarity strength

    return largest_eigenvalue, trace_laplacian, frobenius_norm

def ica_transform(tensor, n_components):
    """
    Apply Independent Component Analysis (ICA) to the input tensor.

    Parameters:
    - tensor (np.ndarray): Input tensor of shape (samples, features).
    - n_components (int): Number of independent components to extract.

    Returns:
    - ica_transformed (np.ndarray): ICA-transformed tensor.
    """
    ica = FastICA(n_components=n_components, random_state=0)
    ica_transformed = ica.fit_transform(tensor)
    return ica_transformed

def compute_cosine_similarity(vec1, vec2):
    """
    Compute the cosine similarity between two vectors.

    Parameters:
    - vec1 (np.ndarray): First vector.
    - vec2 (np.ndarray): Second vector.

    Returns:
    - cosine_similarity (float): Cosine similarity between vec1 and vec2.
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cosine_similarity = dot_product / norm_product
    return cosine_similarity



def compute_diem_torch(A: torch.Tensor, B: torch.Tensor, weight_samples=None):
    """
    Computes the Dimension Insensitive Euclidean Metric (DIEM) for two matrices using PyTorch.

    :param A: First matrix (e.g., pre-trained model parameters) [torch.Tensor]
    :param B: Second matrix (e.g., fine-tuned model parameters) [torch.Tensor]
    :param weight_samples: Optional list of (A, B) weight matrix pairs for empirical expected distance estimation.
    :return: DIEM score [float]
    """
    # Ensure tensors are on the same device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon=1e-6
    device = A.device
    B = B.to(device)

    # Step 1: Compute Frobenius norm (Euclidean distance)
    d = torch.norm(A - B, p='fro')  # Computes ||A - B||_F

    # Step 2: Compute expected Euclidean distance
    n = A.numel()  # Total number of elements (e.g., 4096 * 4096)

    if weight_samples:
        weight_samples = [(Wa.to(device), Wb.to(device)) for Wa, Wb in weight_samples]
        # expected_d = torch.mean(torch.tensor([torch.norm(Wa - Wb, p='fro') for Wa, Wb in weight_samples], device=device))
        distances = torch.tensor([torch.norm(Wa - Wb, p='fro') for Wa, Wb in weight_samples], device=device)
        expected_d = torch.mean(distances)  # Empirical expected distance
    
    else:
        # Theoretical expected distance from DIEM paper
        expected_d = torch.sqrt(torch.tensor(2 * n, dtype=torch.float32, device=device))

    # Step 3: Compute variance of Euclidean distance
    if weight_samples:
        # distances = torch.var(torch.tensor([torch.norm(Wa - Wb, p='fro') for Wa, Wb in weight_samples], device=device))
        variance_d = torch.var(distances) + epsilon

    else:
        # Approximation from DIEM paper
        variance_d = torch.tensor(2 * n * (1 - 1 / torch.pi), dtype=torch.float32, device=device)  # <-- FIXED

    # Step 4: Compute max possible Euclidean distance (v_M)
    A_min, A_max = torch.min(A), torch.max(A)
    v_M = torch.sqrt(torch.tensor(n, dtype=torch.float32, device=device)) * (A_max - A_min)
    # print("v_M", v_M)
    # Step 5: Compute min possible Euclidean distance (v_m)
    v_m = torch.tensor(0.0, device=device)  # Min distance is 0 when A == B
    
    # Step 6: Compute the DIEM score before scaling
    diem_unscaled = (d - expected_d) / torch.sqrt(variance_d)  # Standardized DIEM
    # print("torch.sqrt(variance_d) ", torch.sqrt(variance_d) )
    # print("expected_d", expected_d)
    # print("diem_unscaled", diem_unscaled)
    # Step 7: Apply max-min scaling
    if v_M - v_m == 0:  # Prevent division by zero
        return 0
    diem_scaled = ((diem_unscaled - v_m) / (v_M - v_m)) * (v_M - v_m)

    return diem_scaled.item()  # Convert to Python float



def compute_diem_torch_iqr(A: torch.Tensor, B: torch.Tensor, weight_samples=None, epsilon=1e-6):
    """
    Computes the DIEM score using Interquartile Range (IQR) scaling instead of standard deviation,
    with IQR-based max-min scaling.
    
    :param A: Pre-trained model layer weights (torch.Tensor)
    :param B: Fine-tuned model layer weights (torch.Tensor)
    :param weight_samples: Optional list of (A, B) samples for empirical expected distance.
    :param epsilon: Small constant to prevent division by zero.
    :return: DIEM score [float] or None if unchanged.
    """
    # Ensure tensors are on the same device (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon=1e-6

    device = A.device
    B = B.to(device)

    # **Check if the layer is unchanged (A == B)**
    if torch.equal(A, B):
        return None  # Skip unchanged layers

    # Step 1: Compute Frobenius norm (Euclidean distance)
    d = torch.norm(A - B, p='fro') + epsilon  # ✅ Avoids exact 0 values

    # Step 2: Compute expected Euclidean distance
    n = A.numel()  # Total number of elements (e.g., 4096 * 4096)

    if weight_samples:
        weight_samples = [(Wa.to(device), Wb.to(device)) for Wa, Wb in weight_samples]

        # Compute Frobenius norm distances
        distances = torch.tensor([torch.norm(Wa - Wb, p='fro') for Wa, Wb in weight_samples], device=device)

        # Compute expected Euclidean distance (empirical mean)
        expected_d = torch.mean(distances)
    else:
        # Theoretical expected distance from DIEM paper
        expected_d = torch.sqrt(torch.tensor(2 * n, dtype=torch.float32, device=device))

    # Step 3: Compute IQR-based scaling factor
    if weight_samples:
        q1 = torch.quantile(distances, 0.25)  # 25th percentile (Q1)
        q3 = torch.quantile(distances, 0.75)  # 75th percentile (Q3)
        iqr_scale = q3 - q1 + epsilon  # ✅ IQR-based normalization factor
    else:
        iqr_scale = torch.tensor(2 * n * (1 - 1 / torch.pi), dtype=torch.float32, device=device) + epsilon  

    # Step 4: Compute IQR-Based Max-Min Scaling (`v_m` and `v_M`)
    A_q1 = torch.quantile(A, 0.25)
    A_q3 = torch.quantile(A, 0.75)
    B_q1 = torch.quantile(B, 0.25)

    v_M = A_q3 - A_q1  # ✅ IQR-based max distance
    v_m = A_q1 - B_q1  # ✅ IQR-based min distance (instead of 0)

    # Step 5: Compute the DIEM score before scaling
    diem_unscaled = (d - expected_d) / iqr_scale  # ✅ Now using IQR-based normalization

    # Step 6: Apply IQR-based max-min scaling
    if v_M - v_m == 0:  # Prevent division by zero
        return None  # Skip unchanged layers
    diem_scaled = ((diem_unscaled - v_m) / (v_M - v_m)) * (v_M - v_m)

    return diem_scaled.item()  # Convert to Python float

