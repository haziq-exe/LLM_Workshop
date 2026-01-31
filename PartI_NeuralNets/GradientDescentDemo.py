import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GradientDescentDemo:
    def __init__(self, START_X, START_Y, LEARNING_RATE, NUM_STEPS):

        x_range = np.linspace(-4, 4, 100)
        y_range = np.linspace(-4, 4, 100)
        X, Y = np.meshgrid(x_range, y_range)
        Z = self.complex_landscape(X, Y)

        path_x, path_y, path_z = self.gradient_descent_path(START_X, START_Y, LEARNING_RATE, NUM_STEPS)

        fig = plt.figure(figsize=(14, 5))

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
        ax1.plot(path_x, path_y, path_z, 'r-', linewidth=2, label='Path')
        ax1.scatter([path_x[0]], [path_y[0]], [path_z[0]], color='green', s=100, label='Start')
        ax1.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], color='red', s=100, label='End')
        ax1.set_xlabel('Weight 1')
        ax1.set_ylabel('Weight 2')
        ax1.set_zlabel('Loss')
        ax1.set_title('Gradient Descent in 3D')
        ax1.legend()

        # Top-down view
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
        ax2.plot(path_x, path_y, 'r-', linewidth=2, label='Path')
        ax2.scatter([path_x[0]], [path_y[0]], color='green', s=100, label='Start', zorder=5)
        ax2.scatter([path_x[-1]], [path_y[-1]], color='red', s=100, label='End', zorder=5)
        ax2.set_xlabel('Weight 1')
        ax2.set_ylabel('Weight 2')
        ax2.set_title('Top-Down View (Contour Map)')
        ax2.legend()
        plt.colorbar(contour, ax=ax2, label='Loss')

        plt.tight_layout()
        plt.show()

        print(f"Starting loss: {path_z[0]:.4f}")
        print(f"Final loss: {path_z[-1]:.4f}")

    def complex_landscape(self, x, y):
        """A hilly 3D landscape with multiple local minima"""
        return (np.sin(x) * np.cos(y) * 3 + 
                0.3 * (x**2 + y**2) + 
                np.sin(2*x) * np.cos(2*y) * 1.5)

    def gradient(self, x, y, epsilon=0.01):
        """Compute gradient (partial derivatives) at point (x, y)"""
        dx = (self.complex_landscape(x + epsilon, y) - self.complex_landscape(x - epsilon, y)) / (2 * epsilon)
        dy = (self.complex_landscape(x, y + epsilon) - self.complex_landscape(x, y - epsilon)) / (2 * epsilon)
        return dx, dy

    def gradient_descent_path(self, start_x, start_y, learning_rate=0.1, num_steps=100):
        """Perform gradient descent and return the path"""
        x, y = start_x, start_y
        path_x, path_y, path_z = [x], [y], [self.complex_landscape(x, y)]
        
        for _ in range(num_steps):
            dx, dy = self.gradient(x, y)
            x = x - learning_rate * dx
            y = y - learning_rate * dy
            path_x.append(x)
            path_y.append(y)
            path_z.append(self.complex_landscape(x, y))
        
        return path_x, path_y, path_z