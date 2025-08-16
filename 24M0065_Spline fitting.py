import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

# Airfoil data points arranged horizontally in X,Y pairs
airfoil_data = [100,0, 95.023,0.881, 90.049,1.739, 85.07,2.618, 80.084,3.492, 75.089,4.344, 
                70.087,5.153, 65.076,5.899, 60.057,6.562, 55.031,7.125, 50,7.567, 44.964,7.894, 
                39.924,8.062, 34.882,8.059, 29.84,7.872, 24.8,7.499, 19.765,6.929, 14.735,6.138, 
                9.718,5.063, 7.218,4.379, 4.727,3.544, 2.257,2.46, 1.041,1.719, 0.567,1.32, 
                0.336,1.071, 0,0, 0.664,-0.871, 0.933,-1.04, 1.459,-1.291, 2.743,-1.716, 
                5.273,-2.28, 7.782,-2.685, 10.282,-2.995, 15.265,-3.446, 20.235,-3.745, 
                25.2,-3.919, 30.16,-3.984, 35.1118,-3.939, 40.076,-3.778, 45.035,-3.514, 
                50,-3.164, 54.969,-2.745, 59.943,-2.278, 64.924,-1.799, 69.913,-1.265, 
                74.911,-0.764, 79.916,-0.308, 84.93,0.074, 89.951,0.329, 94.977,0.33, 100,0]

def load_airfoil():
    # Converting the flat list into X,Y pairs
    coords = []
    for i in range(0, len(airfoil_data), 2):
        x = airfoil_data[i]
        y = airfoil_data[i+1]
        coords.append((x, y))
    
    if not coords:
        raise ValueError("No valid coordinate data found in airfoil section.")

    x, y = zip(*coords)
    return np.array(x), np.array(y)

# Performing spline fitting
def spline_fit(x, y, num_points=300):
    tck, u = splprep([x, y], s=0)
    unew = np.linspace(0, 1, num_points)
    x_smooth, y_smooth = splev(unew, tck)
    return x_smooth, y_smooth, tck

# Plotting airfoil
def plot_airfoil(x, y, x_smooth, y_smooth):
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, 'o', label='Original Data', markersize=3)
    plt.plot(x_smooth, y_smooth, '-', label='Spline Fit', linewidth=2)
    plt.axis('equal')
    plt.title('Spline Fit of NACA 63-412 Airfoil')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x, y = load_airfoil()
    x_smooth, y_smooth, spline = spline_fit(x, y)
    plot_airfoil(x, y, x_smooth, y_smooth)