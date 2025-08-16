import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Airfoil data points arranged horizontally in X,Y pairs (in mm)
airfoil_data = [100,0, 95.023,0.881, 90.049,1.739, 85.07,2.618, 80.084,3.492, 75.089,4.344, 
                70.087,5.153, 65.076,5.899, 60.057,6.562, 55.031,7.125, 50,7.567, 44.964,7.894, 
                39.924,8.062, 34.882,8.059, 29.84,7.872, 24.8,7.499, 19.765,6.929, 14.735,6.138, 
                9.718,5.063, 7.218,4.379, 4.727,3.544, 2.257,2.46, 1.041,1.719, 0.567,1.32, 
                0.336,1.071, 0,0, 0.664,-0.871, 0.933,-1.04, 1.459,-1.291, 2.743,-1.716, 
                5.273,-2.28, 7.782,-2.685, 10.282,-2.995, 15.265,-3.446, 20.235,-3.745, 
                25.2,-3.919, 30.16,-3.984, 35.1118,-3.939, 40.076,-3.778, 45.035,-3.514, 
                50,-3.164, 54.969,-2.745, 59.943,-2.278, 64.924,-1.799, 69.913,-1.265, 
                74.911,-0.764, 79.916,-0.308, 84.93,0.074, 89.951,0.329, 94.977,0.33, 100,0]

# === Step 1: Getting airfoil data from embedded coordinates ===
def get_airfoil():
    # Converting the flat list into X,Y pairs and convert to meters
    coords = []
    for i in range(0, len(airfoil_data), 2):
        x = airfoil_data[i] / 1000  # Convert mm to meters
        y = airfoil_data[i+1] / 1000
        coords.append([x, y])
    
    return np.array(coords)

# === Step 2: Resampling airfoil with periodic spline ===
def resample_airfoil(coords, n_points):
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack((coords, coords[0]))

    s = np.zeros(len(coords))
    for i in range(1, len(coords)):
        s[i] = s[i-1] + np.linalg.norm(coords[i] - coords[i-1])
    s /= s[-1]

    fx = interpolate.CubicSpline(s, coords[:, 0], bc_type='periodic')
    fy = interpolate.CubicSpline(s, coords[:, 1], bc_type='periodic')

    s_new = np.linspace(0, 1, n_points, endpoint=False)
    x = fx(s_new)
    y = fy(s_new)

    return np.vstack((x, y)).T

# === Step 3: Generating outer boundary (rectangle) ===
def rectangular_boundary(n_points, width=10.0, height=10.0):
    n_per_side = n_points // 4
    remainder = n_points % 4

    x_right = np.ones(n_per_side + remainder) * width
    y_right = np.linspace(-height, height, n_per_side + remainder)

    x_top = np.linspace(width, -width, n_per_side)
    y_top = np.ones(n_per_side) * height

    x_left = np.ones(n_per_side) * -width
    y_left = np.linspace(height, -height, n_per_side)

    x_bottom = np.linspace(-width, width, n_per_side)
    y_bottom = np.ones(n_per_side) * -height

    x = np.concatenate([x_right, x_top, x_left, x_bottom])
    y = np.concatenate([y_right, y_top, y_left, y_bottom])

    return np.vstack((x, y)).T

# === Step 4: TFI Grid Generation ===
def generate_tfi_grid(inner, outer, n_radial):
    grid_x = np.zeros((n_radial, len(inner)))
    grid_y = np.zeros((n_radial, len(inner)))

    for i in range(n_radial):
        t = (i / (n_radial - 1))**2  # Quadratic clustering
        grid_x[i, :] = (1 - t) * inner[:, 0] + t * outer[:, 0]
        grid_y[i, :] = (1 - t) * inner[:, 1] + t * outer[:, 1]

    return grid_x, grid_y

# === Step 5: Elliptic Grid Smoothing (Fixed Periodic BCs) ===
def smooth_elliptic(grid_x, grid_y, n_iter=100, alpha=0.1, beta=0.1):
    ni, nj = grid_x.shape
    for _ in range(n_iter):
        x_old, y_old = grid_x.copy(), grid_y.copy()

        for i in range(1, ni-1):
            for j in range(nj):
                j_prev = (j - 1) % nj  # Periodic BC in circumferential direction
                j_next = (j + 1) % nj

                # Metric terms
                x_xi = (x_old[i+1, j] - x_old[i-1, j]) / 2
                x_eta = (x_old[i, j_next] - x_old[i, j_prev]) / 2
                y_xi = (y_old[i+1, j] - y_old[i-1, j]) / 2
                y_eta = (y_old[i, j_next] - y_old[i, j_prev]) / 2

                alpha = x_eta**2 + y_eta**2
                beta = x_xi * x_eta + y_xi * y_eta
                gamma = x_xi**2 + y_xi**2

                # Laplacian smoothing (with cross-derivative term)
                grid_x[i, j] = (alpha * (x_old[i+1, j] + x_old[i-1, j]) +
                               gamma * (x_old[i, j_next] + x_old[i, j_prev]) -
                               0.5 * beta * (x_old[i+1, j_next] - x_old[i+1, j_prev] -
                                             x_old[i-1, j_next] + x_old[i-1, j_prev])) / (2 * (alpha + gamma))

                grid_y[i, j] = (alpha * (y_old[i+1, j] + y_old[i-1, j]) +
                               gamma * (y_old[i, j_next] + y_old[i, j_prev]) -
                               0.5 * beta * (y_old[i+1, j_next] - y_old[i+1, j_prev] -
                                             y_old[i-1, j_next] + y_old[i-1, j_prev])) / (2 * (alpha + gamma))

        # Fix boundaries (Dirichlet conditions)
        grid_x[0, :] = x_old[0, :]  # Airfoil (fixed)
        grid_y[0, :] = y_old[0, :]
        grid_x[-1, :] = x_old[-1, :]  # Outer boundary (fixed)
        grid_y[-1, :] = y_old[-1, :]

    return grid_x, grid_y

# === Step 6: Plotting ===
def plot_grid(grid_x, grid_y, title):
    plt.figure(figsize=(10, 8))
    ni, nj = grid_x.shape

    # Plotting grid lines (ensure periodic closure)
    for i in range(ni):
        plt.plot(np.append(grid_x[i, :], grid_x[i, 0]), 
                 np.append(grid_y[i, :], grid_y[i, 0]), 'b-', lw=0.5, alpha=0.7)
    for j in range(nj):
        plt.plot(grid_x[:, j], grid_y[:, j], 'b-', lw=0.5, alpha=0.7)

    # Plotting airfoil (closed loop)
    plt.plot(np.append(grid_x[0, :], grid_x[0, 0]), 
             np.append(grid_y[0, :], grid_y[0, 0]), 'r-', lw=2.0)

    plt.xlim(-11, 11)
    plt.ylim(-11, 11)
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(False)
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    # Generating initial TFI grid
    airfoil_coords = get_airfoil()
    airfoil_resampled = resample_airfoil(airfoil_coords, 40)
    outer_boundary = rectangular_boundary(40, width=10.0, height=10.0)
    grid_x_tfi, grid_y_tfi = generate_tfi_grid(airfoil_resampled, outer_boundary, 40)

    # Smoothing using elliptic grid generation
    grid_x_smooth, grid_y_smooth = smooth_elliptic(grid_x_tfi, grid_y_tfi, n_iter=100)

    # Plotting results
    plot_grid(grid_x_tfi, grid_y_tfi, "Initial 40x40 TFI Grid (Before Smoothing)")
    plot_grid(grid_x_smooth, grid_y_smooth, "40x40 Grid After Elliptic Smoothing (Fixed)")