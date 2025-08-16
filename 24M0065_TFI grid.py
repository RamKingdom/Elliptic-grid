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
    # Convert the flat list into X,Y pairs and convert to meters
    coords = []
    for i in range(0, len(airfoil_data), 2):
        x = airfoil_data[i] / 1000  # Convert mm to meters
        y = airfoil_data[i+1] / 1000
        coords.append([x, y])
    
    return np.array(coords)

# === Step 2: Uniformly resampling airfoil for inner boundary ===
def resample_airfoil(coords, n):
    # Ensure it's a closed loop
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack((coords, coords[0]))

    # Computing cumulative arc length
    s = np.zeros(len(coords))
    for i in range(1, len(coords)):
        s[i] = s[i - 1] + np.linalg.norm(coords[i] - coords[i - 1])
    s /= s[-1]  # normalizing

    # Fitting splines
    fx = interpolate.CubicSpline(s, coords[:, 0], bc_type='periodic')
    fy = interpolate.CubicSpline(s, coords[:, 1], bc_type='periodic')

    s_uniform = np.linspace(0, 1, n, endpoint=False)
    x = fx(s_uniform)
    y = fy(s_uniform)

    return np.vstack((x, y)).T

# === Step 3: Defining outer boundary (rectangle with rounded corners) ===
def rectangular_boundary(n, width=10.0, height=10.0):
    # Creating a rectangle with rounded corners
    # Distributing points evenly around the rectangle
    n_per_side = n // 4
    remainder = n % 4
    
    # Right side
    x_right = np.ones(n_per_side + remainder) * width
    y_right = np.linspace(-height, height, n_per_side + remainder)
    
    # Top side
    x_top = np.linspace(width, -width, n_per_side)
    y_top = np.ones(n_per_side) * height
    
    # Left side
    x_left = np.ones(n_per_side) * -width
    y_left = np.linspace(height, -height, n_per_side)
    
    # Bottom side
    x_bottom = np.linspace(-width, width, n_per_side)
    y_bottom = np.ones(n_per_side) * -height
    
    # Combine all sides
    x = np.concatenate([x_right, x_top, x_left, x_bottom])
    y = np.concatenate([y_right, y_top, y_left, y_bottom])
    
    return np.vstack((x, y)).T

# === Step 4: TFI Grid Generation ===
def generate_tfi_grid(inner, outer, n_radial):
    grid_x = np.zeros((n_radial, len(inner)))
    grid_y = np.zeros((n_radial, len(inner)))

    for i in range(n_radial):
        t = i / (n_radial - 1)
        # Using a non-linear distribution to cluster points near airfoil
        t = t**2  # Quadratic distribution (more points near airfoil)
        curve = (1 - t) * inner + t * outer
        grid_x[i, :] = curve[:, 0]
        grid_y[i, :] = curve[:, 1]

    return grid_x, grid_y

# === Step 5: Plotting ===
def plot_grid(grid_x, grid_y):
    plt.figure(figsize=(10, 10))
    n, m = grid_x.shape
    
    # Plotting grid lines
    for i in range(n):
        plt.plot(grid_x[i, :], grid_y[i, :], 'b', lw=0.4, alpha=0.6)
    for j in range(m):
        plt.plot(grid_x[:, j], grid_y[:, j], 'b', lw=0.4, alpha=0.6)
    
    # Plotting airfoil in red for visibility
    plt.plot(grid_x[0, :], grid_y[0, :], 'r-', lw=2.0)
    
    # Setting axis limits to show the full domain
    plt.xlim(-10.5, 10.5)
    plt.ylim(-10.5, 10.5)
    
    plt.axis('equal')
    plt.title("40x40 O-type TFI Grid around NACA 63-412\n(Far field boundary at 10m from leading edge)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.grid(False)
    plt.show()

# === Main Execution ===
if __name__ == "__main__":
    coords = get_airfoil()
    n_circumferential = 40
    n_radial = 40

    airfoil_loop = resample_airfoil(coords, n_circumferential)
    outer_loop = rectangular_boundary(n_circumferential, width=10, height=10)

    grid_x, grid_y = generate_tfi_grid(airfoil_loop, outer_loop, n_radial)
    plot_grid(grid_x, grid_y)