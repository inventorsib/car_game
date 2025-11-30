import heapq
from typing import Tuple, List
import pygame
import math
import sys
import random
import numpy as np

# Initialize Pygame
pygame.init()
# Screen settings
WIDTH, HEIGHT = 1300, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Top-Down Car - Bicycle Model with Wheels + A* Pathfinding")
# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
LIGHT_GREEN = (144, 238, 144)

# A* Node class
class Node:
    def __init__(self, x, y):
        self.x = x 
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.neighbors = []
        self.previous = None
        self.obstacle = False
        self.in_open_set = False
        self.in_closed_set = False

    def add_neighbors(self, grid, columns, rows):
        neighbor_x = self.x
        neighbor_y = self.y
    
        # 4-directional movement
        if neighbor_x < columns - 1:
            self.neighbors.append(grid[neighbor_x+1][neighbor_y])
        if neighbor_x > 0:
            self.neighbors.append(grid[neighbor_x-1][neighbor_y])
        if neighbor_y < rows - 1:
            self.neighbors.append(grid[neighbor_x][neighbor_y + 1])
        if neighbor_y > 0: 
            self.neighbors.append(grid[neighbor_x][neighbor_y - 1])
        
        # 8-directional movement (diagonals)
        if neighbor_x > 0 and neighbor_y > 0:
            self.neighbors.append(grid[neighbor_x-1][neighbor_y-1])
        if neighbor_x < columns - 1 and neighbor_y > 0:
            self.neighbors.append(grid[neighbor_x+1][neighbor_y-1])
        if neighbor_x > 0 and neighbor_y < rows - 1:
            self.neighbors.append(grid[neighbor_x-1][neighbor_y+1])
        if neighbor_x < columns - 1 and neighbor_y < rows - 1:
            self.neighbors.append(grid[neighbor_x+1][neighbor_y+1])

    def __lt__(self, other):
        return self.f < other.f

# A* Pathfinder class
class AStarPathfinder:
    def __init__(self, grid_size=20, safety_margin=2):
        self.grid_size = grid_size
        self.cols = WIDTH // grid_size
        self.rows = HEIGHT // grid_size
        self.grid = None
        self.path = []
        self.safety_margin = safety_margin  # Количество дополнительных клеток запаса
        
    def mark_obstacle_in_grid(self, obstacle):
        """Mark obstacle area in grid with safety margin"""
        if obstacle.type == "rectangle":
            # Convert obstacle rectangle to grid coordinates with safety margin
            left = max(0, int((obstacle.x - obstacle.width//2) / self.grid_size) - self.safety_margin)
            right = min(self.cols - 1, int((obstacle.x + obstacle.width//2) / self.grid_size) + self.safety_margin)
            top = max(0, int((obstacle.y - obstacle.height//2) / self.grid_size) - self.safety_margin)
            bottom = min(self.rows - 1, int((obstacle.y + obstacle.height//2) / self.grid_size) + self.safety_margin)
            
            for i in range(left, right + 1):
                for j in range(top, bottom + 1):
                    self.grid[i][j].obstacle = True
                    
        else:  # circle
            center_x = obstacle.x / self.grid_size
            center_y = obstacle.y / self.grid_size
            radius = obstacle.radius / self.grid_size + self.safety_margin  # Увеличиваем радиус
            
            left = max(0, int(center_x - radius))
            right = min(self.cols - 1, int(center_x + radius))
            top = max(0, int(center_y - radius))
            bottom = min(self.rows - 1, int(center_y + radius))
            
            for i in range(left, right + 1):
                for j in range(top, bottom + 1):
                    dist = math.sqrt((i - center_x)**2 + (j - center_y)**2)
                    if dist <= radius:
                        self.grid[i][j].obstacle = True
        
    def create_grid(self):
        """Create grid based on current obstacles"""
        self.grid = []
        for i in range(self.cols):
            self.grid.append([])
            for j in range(self.rows):
                self.grid[-1].append(Node(i, j))
        return self.grid
    
    def update_grid_with_obstacles(self, obstacles):
        """Update grid with obstacle information"""
        if self.grid is None:
            self.create_grid()
            
        # Reset grid
        for i in range(self.cols):
            for j in range(self.rows):
                self.grid[i][j].obstacle = False
                self.grid[i][j].neighbors = []
        
        # Mark obstacles
        for obstacle in obstacles:
            self.mark_obstacle_in_grid(obstacle)
            
        # Update neighbors
        for i in range(self.cols):
            for j in range(self.rows):
                self.grid[i][j].add_neighbors(self.grid, self.cols, self.rows)
                
        return self.grid
    
    
    def world_to_grid(self, world_x, world_y):
        """Convert world coordinates to grid coordinates"""
        grid_x = int(world_x / self.grid_size)
        grid_y = int(world_y / self.grid_size)
        return max(0, min(self.cols - 1, grid_x)), max(0, min(self.rows - 1, grid_y))
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates"""
        world_x = grid_x * self.grid_size + self.grid_size // 2
        world_y = grid_y * self.grid_size + self.grid_size // 2
        return world_x, world_y
    
    def find_path(self, start_world, end_world, obstacles):
        """Find path from start to end using A*"""
        # Update grid with current obstacles
        self.update_grid_with_obstacles(obstacles)
        
        # Convert world coordinates to grid coordinates
        start_grid = self.world_to_grid(start_world[0], start_world[1])
        end_grid = self.world_to_grid(end_world[0], end_world[1])
        
        start_node = self.grid[start_grid[0]][start_grid[1]]
        end_node = self.grid[end_grid[0]][end_grid[1]]
        
        # Check if start or end is in obstacle
        if start_node.obstacle or end_node.obstacle:
            return []
        
        # Initialize sets
        open_set = []
        closed_set = set()
        
        # Reset node states
        for i in range(self.cols):
            for j in range(self.rows):
                self.grid[i][j].g = float('inf')
                self.grid[i][j].f = float('inf')
                self.grid[i][j].previous = None
                self.grid[i][j].in_open_set = False
                self.grid[i][j].in_closed_set = False
        
        # Initialize start node
        start_node.g = 0
        start_node.h = self.heuristic(start_node, end_node)
        start_node.f = start_node.g + start_node.h
        heapq.heappush(open_set, (start_node.f, id(start_node), start_node))
        start_node.in_open_set = True
        
        while open_set:
            current_f, _, current_node = heapq.heappop(open_set)
            current_node.in_open_set = False
            
            if current_node == end_node:
                return self.reconstruct_path(current_node)
            
            current_node.in_closed_set = True
            closed_set.add(current_node)
            
            for neighbor in current_node.neighbors:
                if neighbor.obstacle or neighbor.in_closed_set:
                    continue
                
                tentative_g = current_node.g + self.distance(current_node, neighbor)
                
                if tentative_g < neighbor.g:
                    neighbor.previous = current_node
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, end_node)
                    neighbor.f = neighbor.g + neighbor.h
                    
                    if not neighbor.in_open_set:
                        heapq.heappush(open_set, (neighbor.f, id(neighbor), neighbor))
                        neighbor.in_open_set = True
        
        return []  # No path found
    
    def heuristic(self, node_a, node_b):
        """Manhattan distance heuristic"""
        return abs(node_a.x - node_b.x) + abs(node_a.y - node_b.y)
    
    def distance(self, node_a, node_b):
        """Distance between nodes (1 for adjacent, sqrt(2) for diagonal)"""
        dx = abs(node_a.x - node_b.x)
        dy = abs(node_a.y - node_b.y)
        if dx == 1 and dy == 1:
            return 1#math.sqrt(2)*0.1
        return 1
    
    def reconstruct_path(self, end_node):
        """Reconstruct path from end node to start"""
        path = []
        current = end_node
        while current:
            world_x, world_y = self.grid_to_world(current.x, current.y)
            path.append((world_x, world_y))
            current = current.previous
        return path[::-1]  # Reverse to get start to end
    
    def draw_grid(self, surface):
        """Draw the A* grid for debugging"""
        if self.grid is None:
            return
            
        for i in range(self.cols):
            for j in range(self.rows):
                world_x, world_y = self.grid_to_world(i, j)
                rect = pygame.Rect(world_x - self.grid_size//2, world_y - self.grid_size//2, 
                                 self.grid_size, self.grid_size)
                
                if self.grid[i][j].obstacle:
                    pygame.draw.rect(surface, (200, 100, 100, 128), rect)
                else:
                    pygame.draw.rect(surface, (200, 200, 200, 50), rect, 1)
    
    def draw_path(self, surface, path):
        """Draw the A* path"""
        if len(path) > 1:
            for i in range(len(path) - 1):
                pygame.draw.line(surface, GREEN, path[i], path[i+1], 3)
            
            # Draw path points
            for point in path:
                pygame.draw.circle(surface, LIGHT_GREEN, (int(point[0]), int(point[1])), 4)

# Obstacles (your existing Obstacle class remains the same)
class Obstacle:
    def __init__(self, x, y, obstacle_type="rectangle"):
        self.x = x
        self.y = y
        self.type = obstacle_type
        self.width = random.randint(30, 80)
        self.height = random.randint(30, 80)
        self.radius = random.randint(20, 40)
        self.color = random.choice([GRAY, DARK_GRAY])
       
    def draw(self, surface):
        if self.type == "rectangle":
            pygame.draw.rect(surface, self.color, (self.x - self.width//2, self.y - self.height//2, self.width, self.height))
            pygame.draw.rect(surface, BLACK, (self.x - self.width//2, self.y - self.height//2, self.width, self.height), 2)
        elif self.type == "circle":
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
            pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), self.radius, 2)
   
    def get_rect(self):
        if self.type == "rectangle":
            return pygame.Rect(self.x - self.width//2, self.y - self.height//2, self.width, self.height)
        else:
            return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)
   
    def check_collision(self, car_corners):
        obstacle_rect = self.get_rect()
       
        for corner_x, corner_y in car_corners:
            if obstacle_rect.collidepoint(corner_x, corner_y):
                return True
       
        if self.type == "circle":
            for corner_x, corner_y in car_corners:
                distance = math.sqrt((corner_x - self.x)**2 + (corner_y - self.y)**2)
                if distance <= self.radius:
                    return True
       
        return False

# Controller (modified to use A* path)
class Controller:
    def __init__(self):
        self.trajectory = []
        self.target_trajectory = []
        self.max_trajectory_points = 1500
        self.max_target_trajectory_points = 1500
        self.target_point = None
        self.control_mode = "manual"
        self.follow_speed = 0.1
        self.lookahead_distance = 10
        self.a_star_path = []  # Store A* path
    
    def set_a_star_path(self, path):
        """Set the A* path as target trajectory"""
        self.a_star_path = path
        self.target_trajectory = path.copy()
    
    def add_point(self, x, y):
        self.trajectory.append((x, y))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)
   
    def add_target_point(self, x, y):
        self.target_trajectory.append((x, y))
        if len(self.target_trajectory) > self.max_target_trajectory_points:
            self.target_trajectory.pop(0)

    def draw(self, surface, pathfinder):
        # Draw A* grid and path
        pathfinder.draw_grid(surface)
        if self.a_star_path:
            pathfinder.draw_path(surface, self.a_star_path)
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                alpha = int(255 * i / len(self.trajectory))
                color = (0, 100, 200, alpha)
                pygame.draw.line(surface, color, self.trajectory[i-1], self.trajectory[i], 2)
        
        # Draw target trajectory
        if len(self.target_trajectory) > 1:
            for i in range(1, len(self.target_trajectory)):
                alpha = int(255 * i / len(self.target_trajectory))
                color = (100, 0, 200, alpha)
                pygame.draw.line(surface, color, self.target_trajectory[i-1], self.target_trajectory[i], 2)
       
        if self.target_point:
            pygame.draw.circle(surface, PURPLE, (int(self.target_point[0]), int(self.target_point[1])), 6)
            pygame.draw.circle(surface, WHITE, (int(self.target_point[0]), int(self.target_point[1])), 6, 2)
   
    def find_target_point(self, car_x, car_y, car_angle):
        if len(self.target_trajectory) < 10:
            return None
           
        car_pos = np.array([car_x, car_y])
        lookahead_vector = np.array([math.cos(car_angle), math.sin(car_angle)]) * self.lookahead_distance
        target_area = car_pos + lookahead_vector
       
        min_dist = float('inf')
        target_idx = -1
       
        for i, point in enumerate(self.target_trajectory):
            dist = np.linalg.norm(np.array(point) - target_area)
            if dist < min_dist:
                min_dist = dist
                target_idx = i
       
        if target_idx != -1:
            ahead_idx = min(target_idx + 5, len(self.target_trajectory) - 1)
            return self.target_trajectory[ahead_idx]
       
        return None
    
    def calculate_steering(self, car_x, car_y, car_angle, target_point):
        if target_point is None:
            return 0
           
        car_pos = np.array([car_x, car_y])
        target_pos = np.array(target_point)
       
        to_target = target_pos - car_pos
        target_distance = np.linalg.norm(to_target)
       
        if target_distance < 10:
            return 0
       
        to_target_normalized = to_target / target_distance
        car_direction = np.array([math.cos(car_angle), math.sin(car_angle)])
       
        cross_product = np.cross(car_direction, to_target_normalized)
        dot_product = np.dot(car_direction, to_target_normalized)
        angle_to_target = math.acos(max(min(dot_product, 1), -1))
       
        steering_angle = angle_to_target * np.sign(cross_product)
       
        max_steering = math.pi / 6
        steering_angle = max(min(steering_angle, max_steering), -max_steering)
       
        return steering_angle

# Car class (your existing Car class remains the same)
class Car:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.angle = 0
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.1
        self.deceleration = 0.2
        self.steering_angle = 0
        self.max_steering_angle = math.pi / 6
        self.steering_speed = 0.05
        self.length = 60
        self.width = 30
        self.wheel_radius = 8
        self.wheel_width = 4
       
    def get_corners(self):
        front_offset = self.length * 0.4
        rear_offset = self.length * 0.4
        side_offset = self.width * 0.5
       
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
       
        corners = [
            (self.x + front_offset * cos_angle - side_offset * sin_angle,
             self.y + front_offset * sin_angle + side_offset * cos_angle),
            (self.x + front_offset * cos_angle + side_offset * sin_angle,
             self.y + front_offset * sin_angle - side_offset * cos_angle),
            (self.x - rear_offset * cos_angle + side_offset * sin_angle,
             self.y - rear_offset * sin_angle - side_offset * cos_angle),
            (self.x - rear_offset * cos_angle - side_offset * sin_angle,
             self.y - rear_offset * sin_angle + side_offset * cos_angle)
        ]
        return corners
    
    def update(self):
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
       
        if abs(self.speed) > 0.1:
            turning_radius = self.length / math.tan(self.steering_angle) if self.steering_angle != 0 else float('inf')
            angular_velocity = self.speed / turning_radius if turning_radius != float('inf') else 0
            self.angle += angular_velocity
       
        self.x = max(self.width/2, min(WIDTH - self.width/2, self.x))
        self.y = max(self.length/2, min(HEIGHT - self.length/2, self.y))
       
    def draw_wheel(self, surface, wheel_x, wheel_y, wheel_angle, is_front=False):
        wheel_surface = pygame.Surface((self.wheel_radius * 2 + 2, self.wheel_width), pygame.SRCALPHA)
        pygame.draw.rect(wheel_surface, BLACK, (0, 0, self.wheel_radius * 2, self.wheel_width))
        pygame.draw.line(wheel_surface, WHITE, (0, self.wheel_width // 2), (self.wheel_radius * 2, self.wheel_width // 2), 1)
        rotated_wheel = pygame.transform.rotate(wheel_surface, -math.degrees(wheel_angle))
        wheel_rect = rotated_wheel.get_rect(center=(wheel_x, wheel_y))
        surface.blit(rotated_wheel, wheel_rect)
       
        if is_front and abs(self.steering_angle) > 0.01:
            indicator_length = 15
            end_x = wheel_x + indicator_length * math.cos(wheel_angle)
            end_y = wheel_y + indicator_length * math.sin(wheel_angle)
            pygame.draw.line(surface, RED, (wheel_x, wheel_y), (end_x, end_y), 2)
       
    def draw(self, surface):
        front_offset = self.length * 0.35
        rear_offset = self.length * 0.35
        side_offset = self.width * 0.4
       
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
       
        front_left_x = self.x + front_offset * cos_angle - side_offset * sin_angle
        front_left_y = self.y + front_offset * sin_angle + side_offset * cos_angle
        front_right_x = self.x + front_offset * cos_angle + side_offset * sin_angle
        front_right_y = self.y + front_offset * sin_angle - side_offset * cos_angle
        rear_left_x = self.x - rear_offset * cos_angle - side_offset * sin_angle
        rear_left_y = self.y - rear_offset * sin_angle + side_offset * cos_angle
        rear_right_x = self.x - rear_offset * cos_angle + side_offset * sin_angle
        rear_right_y = self.y - rear_offset * sin_angle - side_offset * cos_angle
       
        car_corners = [
            (front_left_x, front_left_y),
            (front_right_x, front_right_y),
            (rear_right_x, rear_right_y),
            (rear_left_x, rear_left_y)
        ]
        pygame.draw.polygon(surface, RED, car_corners)
       
        front_wheel_angle = self.angle + self.steering_angle
        rear_wheel_angle = self.angle
       
        self.draw_wheel(surface, front_left_x, front_left_y, front_wheel_angle, True)
        self.draw_wheel(surface, front_right_x, front_right_y, front_wheel_angle, True)
        self.draw_wheel(surface, rear_left_x, rear_left_y, rear_wheel_angle)
        self.draw_wheel(surface, rear_right_x, rear_right_y, rear_wheel_angle)
       
        pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), 3)
       
        direction_length = 40
        end_x = self.x + direction_length * cos_angle
        end_y = self.y + direction_length * sin_angle
        pygame.draw.line(surface, BLUE, (self.x, self.y), (end_x, end_y), 2)

def generate_obstacles(count=10):
    obstacles = []
    for _ in range(count):
        obstacle_type = random.choice(["rectangle", "circle"])
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        obstacles.append(Obstacle(x, y, obstacle_type))
    return obstacles

def fill_line_target_trajectory(controller:Controller):
    step = WIDTH/controller.max_target_trajectory_points
    for x in np.arange(0, WIDTH, step):
        y = 400 + 5*np.sin(x*0.01)
        controller.add_target_point(x, y)

# Create game objects
car = Car()
obstacles = generate_obstacles(8)
controller = Controller()
pathfinder = AStarPathfinder(grid_size=25)  # Create A* pathfinder

fill_line_target_trajectory(controller)

# Main game loop
planning_requested = False
goal_pos = None
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                obstacles = generate_obstacles(6)
                car.collision = False
            elif event.key == pygame.K_c:
                controller.trajectory = []
                controller.a_star_path = []
                controller.target_trajectory = []
            elif event.key == pygame.K_t:
                controller.control_mode = "follow" if controller.control_mode == "manual" else "manual"
            elif event.key == pygame.K_p:
                planning_requested = True
            elif event.key == pygame.K_g:
                goal_pos = pygame.mouse.get_pos()
                planning_requested = True
            elif event.key == pygame.K_d:
                # Toggle grid display
                pathfinder.grid_size = 0 if pathfinder.grid_size > 0 else 25
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                goal_pos = event.pos
                planning_requested = True
            elif event.button == 3:  # Right click - set start position
                car.x, car.y = event.pos

    # Path planning with A*
    if planning_requested and goal_pos:
        print(f"Planning path from ({car.x:.1f}, {car.y:.1f}) to ({goal_pos[0]}, {goal_pos[1]})")
        path = pathfinder.find_path((car.x, car.y), goal_pos, obstacles)
        if path:
            controller.set_a_star_path(path)
            print(f"Path found with {len(path)} points")
            controller.control_mode = "follow"
        else:
            print("No path found!")
        planning_requested = False

    # Control handling
    keys = pygame.key.get_pressed()
   
    if controller.control_mode == "manual":
        if keys[pygame.K_UP]:
            car.speed = min(car.speed + car.acceleration, car.max_speed)
        elif keys[pygame.K_DOWN]:
            car.speed = max(car.speed - car.acceleration, -car.max_speed/2)
        else:
            if car.speed > 0:
                car.speed = max(0, car.speed - car.deceleration)
            elif car.speed < 0:
                car.speed = min(0, car.speed + car.deceleration)
       
        if keys[pygame.K_LEFT]:
            car.steering_angle = max(car.steering_angle - car.steering_speed, -car.max_steering_angle)
        elif keys[pygame.K_RIGHT]:
            car.steering_angle = min(car.steering_angle + car.steering_speed, car.max_steering_angle)
        else:
            if car.steering_angle > 0:
                car.steering_angle = max(0, car.steering_angle - car.steering_speed)
            elif car.steering_angle < 0:
                car.steering_angle = min(0, car.steering_angle + car.steering_speed)
    else:
        if abs(car.speed) < 1.0:
            car.speed = 1.0
       
        controller.target_point = controller.find_target_point(car.x, car.y, car.angle)
        if controller.target_point:
            target_steering = controller.calculate_steering(car.x, car.y, car.angle, controller.target_point)
            car.steering_angle = car.steering_angle * 0.7 + target_steering * 0.3

    # Update game state
    car.update()
    if abs(car.speed) > 0.5:
        controller.add_point(car.x, car.y)
    
    car_corners = car.get_corners()
    car.collision = False
    for obstacle in obstacles:
        if obstacle.check_collision(car_corners):
            car.collision = True
            break

    # Drawing
    screen.fill(WHITE)
   
    # Draw grid
    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y), 1)
    
    controller.draw(screen, pathfinder)
    car.draw(screen)
   
    # Draw goal marker
    if goal_pos:
        pygame.draw.circle(screen, ORANGE, (int(goal_pos[0]), int(goal_pos[1])), 8)
        pygame.draw.circle(screen, BLACK, (int(goal_pos[0]), int(goal_pos[1])), 8, 2)

    # Draw game objects
    for obstacle in obstacles:
        obstacle.draw(screen)

    # Display information
    font = pygame.font.SysFont(None, 24)
    speed_text = font.render(f"Speed: {car.speed:.1f}", True, BLACK)
    angle_text = font.render(f"Car Angle: {math.degrees(car.angle):.1f}°", True, BLACK)
    steering_text = font.render(f"Steering Angle: {math.degrees(car.steering_angle):.1f}°", True, BLACK)
    collision_text = font.render(f"Collision: {car.collision}", True, RED if car.collision else BLACK)
    mode_text = font.render(f"Mode: {controller.control_mode.upper()}", True, PURPLE if controller.control_mode == "follow" else BLACK)
    trajectory_text = font.render(f"Trajectory points: {len(controller.trajectory)}", True, BLACK)
    path_text = font.render(f"A* Path points: {len(controller.a_star_path)}", True, GREEN if controller.a_star_path else BLACK)
    position_text = font.render(f"Car Position: {car.x:.1f}, {car.y:.1f}", True, BLACK)
    
    help_text1 = font.render("R: New obstacles, C: Clear, T: Toggle mode, G: Set goal", True, BLACK)
    help_text2 = font.render("Left click: Set goal, Right click: Set start, D: Toggle grid", True, BLACK)

    screen.blit(speed_text, (10, 10))
    screen.blit(angle_text, (10, 40))
    screen.blit(steering_text, (10, 70))
    screen.blit(collision_text, (10, 100))
    screen.blit(mode_text, (10, 130))
    screen.blit(trajectory_text, (10, 160))
    screen.blit(path_text, (10, 190))
    screen.blit(position_text, (10, 220))
    screen.blit(help_text1, (10, HEIGHT - 50))
    screen.blit(help_text2, (10, HEIGHT - 25))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()