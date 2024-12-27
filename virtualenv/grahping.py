import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 

def prepare_coordinate_grids(data):
    """Prepare coordinate grids based on the input data shape."""
    n, m = data.shape
    x = np.arange(m)
    y = np.arange(n)
    return np.meshgrid(x, y)

def contour_plot(data, label):
    X, Y = prepare_coordinate_grids(data)
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, data, levels=20, cmap='viridis')
    plt.colorbar(label=label)
    plt.title(f'Contour Plot of {label}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

def surface_plot(data, label):
    X, Y = prepare_coordinate_grids(data)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, data, cmap='viridis')
    fig.colorbar(surf, label=label)
    ax.set_title(f'3D Surface Plot of {label}')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Probability')
    plt.show()

def scatter_plot(data, label):
    X, Y = prepare_coordinate_grids(data)
    plt.figure(figsize=(10, 8))
    plt.scatter(X.flatten(), Y.flatten(), c=data.flatten(), cmap='viridis', s=50)
    plt.colorbar(label=label)
    plt.title(f'Scatter Plot with Color-Coded {label}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

def vector_field_plot(data, label):
    X, Y = prepare_coordinate_grids(data)
    dx, dy = np.gradient(data)
    plt.figure(figsize=(10, 8))
    plt.quiver(X[::2, ::2], Y[::2, ::2], dx[::2, ::2], dy[::2, ::2], data[::2, ::2], cmap='viridis')
    plt.colorbar(label=label)
    plt.title(f'Vector Field Plot of {label} Gradient')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

def density_plot(data, label):
    X, Y = prepare_coordinate_grids(data)
    plt.figure(figsize=(10, 8))
    plt.pcolormesh(X, Y, data, cmap='viridis', shading='auto')
    plt.colorbar(label=label)
    plt.title(f'Density Plot of {label}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

def heatmap_plot(data, label):
    plt.figure(figsize=(12, 10))
    sns.heatmap(data, cmap='viridis', cbar_kws={'label': label})
    plt.title(f'Heatmap of {label}')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()