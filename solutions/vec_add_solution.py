import numpy as np
from numba import cuda

@cuda.jit
def vec_add(a, b, c):
    i = cuda.grid(1)

    if i < a.shape[0]:
        # Task 1
        # Compute a + b and store the result in c
        c[i] = a[i] + b[i]

size = 1024 * 1024
a_cpu = np.arange(0, size, dtype=np.float32)
b_cpu = a_cpu * 2

a_gpu = cuda.to_device(a_cpu)
b_gpu = cuda.to_device(b_cpu)
c_gpu = cuda.device_array_like(a_cpu)

threads_per_block = 1024
blocks_per_grid = (size + threads_per_block - 1) // threads_per_block
vec_add[blocks_per_grid, threads_per_block](a_gpu, b_gpu, c_gpu)

c_cpu = c_gpu.copy_to_host()

print(c_cpu)
print(np.allclose(a_cpu + b_cpu, c_cpu))