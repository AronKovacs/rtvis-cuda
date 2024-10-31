import pygame
import random

# Parameters for boids
width, height = 1280, 720
n_boids = 256
max_speed = 4
max_force = 0.1

class Boid:
    def __init__(self, x, y):
        self.position = pygame.Vector2(x, y)
        self.velocity = pygame.Vector2(random.uniform(-1, 1), random.uniform(-1, 1)).normalize() * max_speed
        self.acceleration = pygame.Vector2(0, 0)

    def update(self):
        self.velocity += self.acceleration
        self.velocity = self.velocity.normalize() * max_speed
        self.position += self.velocity
        self.acceleration *= 0

    def apply_force(self, force):
        self.acceleration += force

    def flock(self, boids):
        alignment = self.align(boids)
        cohesion = self.cohere(boids)
        separation = self.separate(boids)

        # Weigh the forces
        self.apply_force(alignment)
        self.apply_force(cohesion)
        self.apply_force(separation)

    def align(self, boids):
        perception_radius = 50
        steering = pygame.Vector2(0, 0)
        total = 0
        avg_vector = pygame.Vector2(0, 0)

        for other in boids:
            if other != self and self.position.distance_to(other.position) < perception_radius:
                avg_vector += other.velocity
                total += 1

        if total > 0:
            avg_vector = avg_vector.normalize() * max_speed
            steering = avg_vector - self.velocity
            steering = self.limit(steering, max_force)

        return steering

    def cohere(self, boids):
        perception_radius = 50
        steering = pygame.Vector2(0, 0)
        total = 0
        center_of_mass = pygame.Vector2(0, 0)

        for other in boids:
            if other != self and self.position.distance_to(other.position) < perception_radius:
                center_of_mass += other.position
                total += 1

        if total > 0:
            center_of_mass /= total
            steering = center_of_mass - self.position
            steering = steering.normalize() * max_speed
            steering -= self.velocity
            steering = self.limit(steering, max_force)

        return steering

    def separate(self, boids):
        perception_radius = 25
        steering = pygame.Vector2(0, 0)
        total = 0

        for other in boids:
            distance = self.position.distance_to(other.position)
            if other != self and distance < perception_radius:
                diff = self.position - other.position
                diff /= distance + 0.0001
                steering += diff
                total += 1

        if total > 0 and steering.length() > 0:
            steering = steering.normalize() * max_speed
            steering -= self.velocity
            steering = self.limit(steering, max_force)

        return steering

    def limit(self, vector, max_value):
        if vector.length() > max_value:
            return vector.normalize() * max_value
        return vector

    def edges(self):
        if self.position.x > width:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = width

        if self.position.y > height:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = height

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Boids")
clock = pygame.time.Clock()

# Create boids
boids = [Boid(random.randint(0, width), random.randint(0, height)) for _ in range(n_boids)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 0))

    for boid in boids:
        boid.flock(boids)
        boid.update()
        boid.edges()
        pygame.draw.rect(screen, (255, 255, 255), pygame.Rect((int(boid.position.x), int(boid.position.y)), (2, 2)))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
