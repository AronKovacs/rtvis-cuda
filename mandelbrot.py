import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

# Parameters for the Mandelbrot fractal
width, height = 1024, 1024
min_x, max_x = -2.0, 1.0
min_y, max_y = -1.5, 1.5
max_iters = 1024
escape_radius = 16.0

@cuda.jit
def mandelbrot(image, min_x, max_x, min_y, max_y, max_iters, escape_radius2):
    height, width = image.shape
    x, y = cuda.grid(2)

    if x >= width and y >= height:
        return

    # Task 1
    # Convert x and y to complex plane coordinates
    c_r = 0.0
    c_i = 0.0
    
    # Initialize z and the iteration counter
    z_r = 0.0
    z_i = 0.0
    i = 0

    while z_r * z_r + z_i * z_i <= escape_radius2 and i < max_iters:
        # Task 2
        # Compute z = z^2 + c

        i += 1
    
    # Smooth coloring calculation
    smooth_value = 0.0
    if i < max_iters:
        log_zn = np.log(z_r * z_r + z_i * z_i) / 2
        nu = np.log(log_zn / np.log(2)) / np.log(2)
        smooth_value = i + 1 - nu
    
    image[y, x] = smooth_value

# Allocate image on the GPU (float32 for smooth values)
image_gpu = cuda.device_array((height, width), dtype=np.float32)

# Task 3
# Define block and grid sizes
threads_per_block = (1, 1)
blocks_per_grid_x = image_gpu.shape[1]
blocks_per_grid_y = image_gpu.shape[0]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch the kernel
mandelbrot[blocks_per_grid, threads_per_block](image_gpu, min_x, max_x, min_y, max_y, max_iters, escape_radius * escape_radius)

# Copy result back to host
image_cpu = image_gpu.copy_to_host()

# Display the fractal
image_cpu = np.log(image_cpu + 1)
plt.figure(figsize=(10, 10))
plt.imshow(image_cpu, cmap='inferno', extent=(min_x, max_x, min_y, max_y))
plt.colorbar()
plt.title('Mandelbrot Fractal')
plt.show()