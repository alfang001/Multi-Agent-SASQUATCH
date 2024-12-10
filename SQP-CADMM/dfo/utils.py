import casadi as ca
import numpy as np

# Step 1: define casadi variables
x = ca.SX.sym('x', 2)  # 2x1
a = ca.SX.sym('a', 2)  # 2x1
b = ca.SX.sym('b')

# Step 2: Define the function f(x) = 0.5 * (norm(x - a) - b)^2
f = 0.5 * (ca.norm_2(x - a) - b)**2

# Step 3: Differentiate the function w.r.t x
df_dx = ca.gradient(f, x)

f_derivative = ca.Function('f_derivative', [x, a, b], [df_dx])

def calculate_rmse(estimate, reference):
    """Calculates the RMSE between the estimate and reference.

    Args:
        estimate (np.array): estimate
        reference (np.array): actual

    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean((estimate - reference)**2))

def all_points_within_threshold(points, threshold):
    # Compute the pairwise distances
    pairwise_diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    pairwise_distances = np.sqrt(np.sum(pairwise_diff**2, axis=2))
    
    # Check if all distances are within the threshold
    is_within_threshold = np.all(pairwise_distances <= threshold)
    
    return is_within_threshold