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


