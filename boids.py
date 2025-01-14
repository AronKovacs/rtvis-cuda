import numpy as np
from numba import cuda
import math

# Parameters for boids
mode = 'pygame' # 'pygame' or 'video'
fps = 60
n_video_frames = fps * 10

width, height = 1920, 1080
n_boids = 4000
perception_radius = 50
alignment_weight = 5
cohesion_weight = 2
separation_weight = 100
speed = 4
max_force = 1

if mode == 'pygame':
    import pygame
elif mode == 'video':
    import cv2

@cuda.jit
def boids_step(boids, width, height, perception_radius, alignment_weight, cohesion_weight, separation_weight, speed, max_force):
    n_boids = boids.shape[0]
    eps = np.float32(0.0001)

    i = cuda.grid(1)
    if i >= n_boids:
        return

    boid_x = boids[i * 4]
    boid_y = boids[i * 4 + 1]
    boid_vx = boids[i * 4 + 2]
    boid_vy = boids[i * 4 + 3]

    # Initializing vars for accumulation

    # Alignment
    other_avg_vx = np.float32(0.0)
    other_avg_vy = np.float32(0.0)

    # Cohesion
    center_x = np.float32(0.0)
    center_y = np.float32(0)
    n_seen = np.uint32(0)

    # Separation
    weighted_distance_x = np.float32(0)
    weighted_distance_y = np.float32(0)

    for j in range(n_boids):
        if i == j:
            continue

        other_boid_x = boids[j * 4]
        other_boid_y = boids[j * 4 + 1]
        other_boid_vx = boids[j * 4 + 2]
        other_boid_vy = boids[j * 4 + 3]

        # Task 1
        # Compute the distance to the other boid and skip it if it is outside the perception radius

        # Alignment

        # Task 2
        # Accumulate the velocity of the other seen boids

        # Cohesion

        # Task 3
        # Accumulate what is needed to compute the center of mass of the other seen boids

        # Separation

        # Task 4
        # Accumulate what is needed to avoid the other seen boids

    # Alignment

    # Task 5
    # Compute the average normalized velocity of the other seen boids, multiply it with the weight,
    # and compute how to steer the boid.

    alignment_steering_x = np.float32(0.0)
    alignment_steering_y = np.float32(0.0)

    # Cohesion

    # Task 6
    # Compute the center of mass of the other seen boids and steer the boid
    # so it flies towards it. Also apply the weight.

    cohesion_steering_x = np.float32(0.0)
    cohesion_steering_y = np.float32(0.0)

    # Separation

    # Task 7
    # Steer the boid so it avoids the other seen boids. Do not forget to apply the weight.

    separation_steering_x = np.float32(0.0)
    separation_steering_y = np.float32(0.0)

    # Accumulate forces

    force_x = alignment_steering_x + cohesion_steering_x + separation_steering_x
    force_y = alignment_steering_y + cohesion_steering_y + separation_steering_y

    force_magn = math.sqrt(force_x * force_x + force_y * force_y + eps)
    if force_magn > max_force:
        force_x = max_force * force_x / force_magn
        force_y = max_force * force_y / force_magn

    # Update position and velocity

    boid_vx += force_x
    boid_vy += force_y

    velocity_magn = math.sqrt(boid_vx * boid_vx + boid_vy * boid_vy + eps)
    boid_vx = speed * boid_vx / velocity_magn
    boid_vy = speed * boid_vy / velocity_magn

    boid_x += boid_vx
    boid_y += boid_vy

    # Task 8
    # Handling borders
    # Keep the boids inside the screen

    boids[i * 4] = boid_x
    boids[i * 4 + 1] = boid_y
    boids[i * 4 + 2] = boid_vx
    boids[i * 4 + 3] = boid_vy

boids_arr_cpu = np.zeros((n_boids, 4), np.float32)

# Positions
boids_arr_cpu[:, 0] = np.random.uniform(0, width, n_boids)
boids_arr_cpu[:, 1] = np.random.uniform(0, height, n_boids)

# Velocity
boids_arr_cpu[:, 2] = np.random.uniform(0, 2 * np.pi, n_boids)
boids_arr_cpu[:, 3] = np.sin(boids_arr_cpu[:, 2]) * speed
boids_arr_cpu[:, 2] = np.cos(boids_arr_cpu[:, 2]) * speed

boids_arr_cpu = boids_arr_cpu.reshape(n_boids * 4)

boids_arr_gpu = cuda.to_device(boids_arr_cpu)

if mode == 'pygame':
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Boids")
    clock = pygame.time.Clock()

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        # Task 9
        # Define block and grid sizes
        threads_per_block = 1
        blocks_per_grid = n_boids
        boids_step[blocks_per_grid, threads_per_block](
            boids_arr_gpu,
            np.uint32(width),
            np.uint32(height),
            np.float32(perception_radius),
            np.float32(alignment_weight),
            np.float32(cohesion_weight),
            np.float32(separation_weight),
            np.float32(speed),
            np.float32(max_force))

        boids_arr_gpu.copy_to_host(boids_arr_cpu)
        for i in range(n_boids):
            pygame.draw.rect(screen, (255, 255, 255), pygame.Rect((int(boids_arr_cpu[i * 4]), int(boids_arr_cpu[i * 4 + 1])), (2, 2)))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()
elif mode == 'video':
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))

    for _ in range(n_video_frames):
        # Task 9.1
        # Define block and grid sizes
        # (just copy it from Task 9)
        threads_per_block = 1
        blocks_per_grid = n_boids
        boids_step[blocks_per_grid, threads_per_block](
            boids_arr_gpu,
            np.uint32(width),
            np.uint32(height),
            np.float32(perception_radius),
            np.float32(alignment_weight),
            np.float32(cohesion_weight),
            np.float32(separation_weight),
            np.float32(speed),
            np.float32(max_force))
        
        boids_arr_gpu.copy_to_host(boids_arr_cpu)
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        for i in range(n_boids):
            x = int(boids_arr_cpu[i * 4])
            y = int(boids_arr_cpu[i * 4 + 1])

            for dx in range(2):
                for dy in range(2):
                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue
                    frame[ny, nx] = 255
        
        out.write(frame)

    out.release()