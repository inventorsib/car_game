import pygame
import math
import sys
import random
import numpy as np

import MotionPlanning.HybridAstarPlanner.hybrid_astar as hastar

# Initialize Pygame
pygame.init()
# Screen settings
WIDTH, HEIGHT = 1300, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Top-Down Car - Bicycle Model with Wheels")
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

# Path Planner
class PathPlanner:

    def __init__(self):
        self.planned_path = []
        self.planning_in_progress = False
        self.planning_result = None
       
    def plan_path(self, start_x, start_y, start_yaw, goal_x, goal_y, goal_yaw, obstacles):
        """Plan path using Hybrid A* algorithm"""
        self.planning_in_progress = True
       
        try:
            # Convert obstacles to format for Hybrid A*
            ox, oy = [], []
            for obstacle in obstacles:
                obstacle_points = obstacle.get_hybrid_astar_obstacle_points()
                for point in obstacle_points:
                    ox.append(point[0])
                    oy.append(point[1])
           
            # Add boundary obstacles
            boundary_margin = 20
            for x in range(0, WIDTH, 10):
                ox.append(x)
                oy.append(boundary_margin)
                ox.append(x)
                oy.append(HEIGHT - boundary_margin)
            for y in range(0, HEIGHT, 10):
                ox.append(boundary_margin)
                oy.append(y)
                ox.append(WIDTH - boundary_margin)
                oy.append(y)
           
            # Call Hybrid A* planner
            path = hastar.hybrid_astar_planning(
                start_x, start_y, start_yaw,
                goal_x, goal_y, goal_yaw,
                ox, oy, hastar.C.XY_RESO, hastar.C.YAW_RESO
            )
           
            self.planning_result = path
            if path:
                self.planned_path = list(zip(path.x, path.y))
            else:
                self.planned_path = []
               
        except Exception as e:
            print(f"Path planning error: {e}")
            self.planned_path = []
            self.planning_result = None
           
        self.planning_in_progress = False
        return self.planned_path
   
    def draw(self, surface):
        """Draw planned path"""
        if len(self.planned_path) > 1:
            # Draw the main path
            for i in range(1, len(self.planned_path)):
                start_pos = self.planned_path[i-1]
                end_pos = self.planned_path[i]
                pygame.draw.line(surface, RED, start_pos, end_pos, 3)
           
            # Draw path points
            for point in self.planned_path:
                pygame.draw.circle(surface, BLACK, (int(point[0]), int(point[1])), 3)
           
        # Draw start and goal markers
        if len(self.planned_path) > 0:
            # Start point
            pygame.draw.circle(surface, BLUE, (int(self.planned_path[0][0]), int(self.planned_path[0][1])), 8)
            
            # Goal point with direction indicator
            goal_x, goal_y = self.planned_path[-1]
            pygame.draw.circle(surface, RED, (int(goal_x), int(goal_y)), 8)
            
            # Draw simple direction line from goal
            if len(self.planned_path) > 1:
                prev_x, prev_y = self.planned_path[-2]
                
                # Calculate direction and draw line
                dir_x = goal_x - prev_x
                dir_y = goal_y - prev_y
                length = math.sqrt(dir_x*dir_x + dir_y*dir_y)
                
                if length > 0:
                    # Normalize and scale
                    dir_x, dir_y = dir_x/length * 20, dir_y/length * 20
                    
                    # Draw direction line
                    end_x = goal_x + dir_x
                    end_y = goal_y + dir_y
                    pygame.draw.line(surface, RED, (goal_x, goal_y), (end_x, end_y), 3)


# Obstacles
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
            # Draw border
            pygame.draw.rect(surface, BLACK, (self.x - self.width//2, self.y - self.height//2, self.width, self.height), 2)
        elif self.type == "circle":
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
            pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), self.radius, 2)
   
    def get_hybrid_astar_obstacle_points(self, grid_size=5):
        """Convert obstacle to points for Hybrid A* planning"""
        points = []
        if self.type == "rectangle":
            for x in range(int(self.x - self.width//2), int(self.x + self.width//2), grid_size):
                for y in range(int(self.y - self.height//2), int(self.y + self.height//2), grid_size):
                    points.append((x, y))
        else:  # circle
            for angle in range(0, 360, 10):
                rad = math.radians(angle)
                x = self.x + self.radius * math.cos(rad)
                y = self.y + self.radius * math.sin(rad)
                points.append((x, y))
        return points
    def get_rect(self):
        """Get bounding rectangle for collision detection"""
        if self.type == "rectangle":
            return pygame.Rect(self.x - self.width//2, self.y - self.height//2, self.width, self.height)
        else:  # circle
            return pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)
   
    def check_collision(self, car_corners):
        """Check collision between obstacle and car"""
        obstacle_rect = self.get_rect()
       
        # Check if any car corner is inside obstacle
        for corner_x, corner_y in car_corners:
            if obstacle_rect.collidepoint(corner_x, corner_y):
                return True
       
        # Additional check for circle obstacles
        if self.type == "circle":
            for corner_x, corner_y in car_corners:
                distance = math.sqrt((corner_x - self.x)**2 + (corner_y - self.y)**2)
                if distance <= self.radius:
                    return True
       
        return False

# Controller
class Controller:
    def __init__(self):
        self.trajectory = []  # List of (x, y) points
        self.target_trajectory = []  # List of (x, y) points
        self.max_trajectory_points = 500  # Maximum points to store
        self.max_target_trajectory_points = 500  # Maximum points to store
        self.target_point = None
        self.control_mode = "manual"  # "manual" or "follow"
        self.follow_speed = 0.1
        self.lookahead_distance = 50
    
    def add_point(self, x, y):
        """Add current position to trajectory"""
        self.trajectory.append((x, y))
        # Keep only recent points
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)
   
    def add_target_point(self, x, y):
        """Add point of target trajectory"""
        self.target_trajectory.append((x, y))
        # Keep only recent points
        if len(self.target_trajectory) > self.max_target_trajectory_points:
            self.target_trajectory.pop(0)

    def draw(self, surface):
        """Draw trajectory and target point"""
        # Draw trajectory
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                alpha = int(255 * i / len(self.trajectory))  # Fade effect
                color = (0, 100, 200, alpha)
                start_pos = self.trajectory[i-1]
                end_pos = self.trajectory[i]
                pygame.draw.line(surface, color, start_pos, end_pos, 2)
        if len(self.target_trajectory):
            for i in range(1, len(self.target_trajectory)):
                alpha = int(255 * i / len(self.target_trajectory))  # Fade effect
                color = (100, 0, 200, alpha)
                start_pos = self.target_trajectory[i-1]
                end_pos = self.target_trajectory[i]
                pygame.draw.line(surface, color, start_pos, end_pos, 2)
       
        # Draw target point
        if self.target_point:
            pygame.draw.circle(surface, PURPLE, (int(self.target_point[0]), int(self.target_point[1])), 6)
            pygame.draw.circle(surface, WHITE, (int(self.target_point[0]), int(self.target_point[1])), 6, 2)
   
    def find_target_point(self, car_x, car_y, car_angle):
        """Find target point on trajectory for the car to follow"""
        if len(self.target_trajectory) < 10:
            return None
           
        # Find the point on target trajectory that is lookahead_distance ahead
        car_pos = np.array([car_x, car_y])
        lookahead_vector = np.array([math.cos(car_angle), math.sin(car_angle)]) * self.lookahead_distance
        target_area = car_pos + lookahead_vector
       
        # Find closest point in target trajectory to the target area
        min_dist = float('inf')
        target_idx = -1
       
        for i, point in enumerate(self.target_trajectory):
            dist = np.linalg.norm(np.array(point) - target_area)
            if dist < min_dist:
                min_dist = dist
                target_idx = i
       
        if target_idx != -1:
            # Get a point slightly ahead in the target trajectory
            ahead_idx = min(target_idx + 5, len(self.target_trajectory) - 1)
            return self.target_trajectory[ahead_idx]
       
        return None
    
    def calculate_steering(self, car_x, car_y, car_angle, target_point):
        """Calculate steering angle to reach target point"""
        if target_point is None:
            return 0
           
        car_pos = np.array([car_x, car_y])
        target_pos = np.array(target_point)
       
        # Vector to target
        to_target = target_pos - car_pos
        target_distance = np.linalg.norm(to_target)
       
        if target_distance < 10:  # Too close, don't steer
            return 0
       
        # Normalize vectors
        to_target_normalized = to_target / target_distance
        car_direction = np.array([math.cos(car_angle), math.sin(car_angle)])
       
        # Calculate cross product to determine turn direction
        cross_product = np.cross(car_direction, to_target_normalized)
       
        # Calculate angle between car direction and target direction
        dot_product = np.dot(car_direction, to_target_normalized)
        angle_to_target = math.acos(max(min(dot_product, 1), -1))
       
        # Determine steering angle (positive for right, negative for left)
        steering_angle = angle_to_target * np.sign(cross_product)
       
        # Apply limits
        max_steering = math.pi / 6  # 30 degrees
        steering_angle = max(min(steering_angle, max_steering), -max_steering)
       
        return steering_angle
    
# Car parameters
class Car:
    def __init__(self):
        self.x = WIDTH // 2
        self.y = HEIGHT // 2
        self.angle = 0  # Movement direction (in radians)
        self.speed = 0
        self.max_speed = 5
        self.acceleration = 0.1
        self.deceleration = 0.2
        self.steering_angle = 0  # Steering wheel angle (in radians)
        self.max_steering_angle = math.pi / 6  # Maximum steering angle (45 degrees)
        self.steering_speed = 0.05  # Steering wheel turn speed
        self.length = 60  # Car length
        self.width = 30   # Car width
        self.wheel_radius = 8
        self.wheel_width = 4
       
    def get_corners(self):
        """Get the four corners of the car for collision detection"""
        front_offset = self.length * 0.4
        rear_offset = self.length * 0.4
        side_offset = self.width * 0.5
       
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
       
        corners = [
            # Front left
            (self.x + front_offset * cos_angle - side_offset * sin_angle,
             self.y + front_offset * sin_angle + side_offset * cos_angle),
            # Front right
            (self.x + front_offset * cos_angle + side_offset * sin_angle,
             self.y + front_offset * sin_angle - side_offset * cos_angle),
            # Rear right
            (self.x - rear_offset * cos_angle + side_offset * sin_angle,
             self.y - rear_offset * sin_angle - side_offset * cos_angle),
            # Rear left
            (self.x - rear_offset * cos_angle - side_offset * sin_angle,
             self.y - rear_offset * sin_angle + side_offset * cos_angle)
        ]
        return corners
    def update(self):
        # Update position
        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
       
        # Update movement direction based on steering angle
        if abs(self.speed) > 0.1:  # Turning only possible when moving
            turning_radius = self.length / math.tan(self.steering_angle) if self.steering_angle != 0 else float('inf')
            angular_velocity = self.speed / turning_radius if turning_radius != float('inf') else 0
            self.angle += angular_velocity
       
        # Keep car within screen bounds
        self.x = max(self.width/2, min(WIDTH - self.width/2, self.x))
        self.y = max(self.length/2, min(HEIGHT - self.length/2, self.y))
       
    def draw_wheel(self, surface, wheel_x, wheel_y, wheel_angle, is_front=False):
        """Draw a single wheel with proper rotation"""
        # Create wheel surface
        wheel_surface = pygame.Surface((self.wheel_radius * 2 + 2, self.wheel_width), pygame.SRCALPHA)
       
        # Draw wheel
        pygame.draw.rect(wheel_surface, BLACK,
                        (0, 0, self.wheel_radius * 2, self.wheel_width))
       
        # Draw wheel rim (center line)
        pygame.draw.line(wheel_surface, WHITE,
                        (0, self.wheel_width // 2),
                        (self.wheel_radius * 2, self.wheel_width // 2), 1)
       
        # Rotate wheel
        rotated_wheel = pygame.transform.rotate(wheel_surface, -math.degrees(wheel_angle))
        wheel_rect = rotated_wheel.get_rect(center=(wheel_x, wheel_y))
       
        # Draw wheel on surface
        surface.blit(rotated_wheel, wheel_rect)
       
        # Draw steering angle indicator for front wheels
        if is_front and abs(self.steering_angle) > 0.01:
            indicator_length = 15
            end_x = wheel_x + indicator_length * math.cos(wheel_angle)
            end_y = wheel_y + indicator_length * math.sin(wheel_angle)
            pygame.draw.line(surface, RED, (wheel_x, wheel_y), (end_x, end_y), 2)
       
    def draw(self, surface):
        # Calculate wheel positions relative to car center
        front_offset = self.length * 0.35
        rear_offset = self.length * 0.35
        side_offset = self.width * 0.4
       
        # Calculate global positions of wheels
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)
       
        # Front wheels
        front_left_x = self.x + front_offset * cos_angle - side_offset * sin_angle
        front_left_y = self.y + front_offset * sin_angle + side_offset * cos_angle
       
        front_right_x = self.x + front_offset * cos_angle + side_offset * sin_angle
        front_right_y = self.y + front_offset * sin_angle - side_offset * cos_angle
       
        # Rear wheels
        rear_left_x = self.x - rear_offset * cos_angle - side_offset * sin_angle
        rear_left_y = self.y - rear_offset * sin_angle + side_offset * cos_angle
       
        rear_right_x = self.x - rear_offset * cos_angle + side_offset * sin_angle
        rear_right_y = self.y - rear_offset * sin_angle - side_offset * cos_angle
       
        # Draw car body
        car_corners = [
            (front_left_x, front_left_y),
            (front_right_x, front_right_y),
            (rear_right_x, rear_right_y),
            (rear_left_x, rear_left_y)
        ]
        pygame.draw.polygon(surface, RED, car_corners)
       
        # Draw wheels with proper angles
        # Front wheels turn with steering
        front_wheel_angle = self.angle + self.steering_angle
        # Rear wheels always point straight
        rear_wheel_angle = self.angle
       
        self.draw_wheel(surface, front_left_x, front_left_y, front_wheel_angle, True)
        self.draw_wheel(surface, front_right_x, front_right_y, front_wheel_angle, True)
        self.draw_wheel(surface, rear_left_x, rear_left_y, rear_wheel_angle)
        self.draw_wheel(surface, rear_right_x, rear_right_y, rear_wheel_angle)
       
        # Draw car center and direction
        pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), 3)
       
        # Draw direction line
        direction_length = 40
        end_x = self.x + direction_length * cos_angle
        end_y = self.y + direction_length * sin_angle
        pygame.draw.line(surface, BLUE, (self.x, self.y), (end_x, end_y), 2)

def generate_obstacles(count=10):
    """Generate random obstacles"""
    obstacles = []
    for _ in range(count):
        obstacle_type = random.choice(["rectangle", "circle"])
        x = random.randint(50, WIDTH - 50)
        y = random.randint(50, HEIGHT - 50)
        #obstacles.append(Obstacle(x, y, obstacle_type))
        obstacles.append(Obstacle(x, y))
    return obstacles
def fill_line_target_trajectory(controller:Controller):
    step = WIDTH/controller.max_target_trajectory_points
    for x in np.arange(0, WIDTH, step):
        y = 400 + 50*np.sin(x*0.01)
        controller.add_target_point(x, y)

# Create car, obstacles, controller and path_planner
car = Car()
obstacles = generate_obstacles(8)
controller = Controller()
path_planner = PathPlanner()

fill_line_target_trajectory(controller)



# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                obstacles = generate_obstacles(6)
                car.collision = False
            elif event.key == pygame.K_c:
                controller.trajectory = []  # Clear trajectory
            elif event.key == pygame.K_t:
                # Toggle control mode
                controller.control_mode = "follow" if controller.control_mode == "manual" else "manual"
    # Get key states
    keys = pygame.key.get_pressed()
   
    # Speed control
    if keys[pygame.K_UP]:
        car.speed = min(car.speed + car.acceleration, car.max_speed)
    elif keys[pygame.K_DOWN]:
        car.speed = max(car.speed - car.acceleration, -car.max_speed/2)
    else:
        # Gradual deceleration
        if car.speed > 0:
            car.speed = max(0, car.speed - car.deceleration)
        elif car.speed < 0:
            car.speed = min(0, car.speed + car.deceleration)
    
    # Steering control
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
    # Autonomous control
    else:
        # Maintain constant speed in autonomous mode
        if abs(car.speed) < 2.0:
            car.speed = 2.0
       
        # TODO: control of steering speed
        # Find target point and calculate steering
        controller.target_point = controller.find_target_point(car.x, car.y, car.angle)
        if controller.target_point:
            target_steering = controller.calculate_steering(car.x, car.y, car.angle, controller.target_point)
            # Smooth steering transition
            car.steering_angle = car.steering_angle * 0.7 + target_steering * 0.3
   
    goal_x, goal_y, goal_yaw = 500, 500, 0
    #! Path planing
    path_planner.plan_path(
                        car.x, car.y, car.angle,
                        goal_x, goal_y, goal_yaw,
                        obstacles
                    )

    #! Updating
    # Update car state
    car.update()
    # Add current position to trajectory (only when moving significantly)
    if abs(car.speed) > 0.5:
        controller.add_point(car.x, car.y)
    # Check collisions
    car_corners = car.get_corners()
    car.collision = False
    for obstacle in obstacles:
        if obstacle.check_collision(car_corners):
            car.collision = True
            break
   
    #! Drawing
    screen.fill(WHITE)
   
    # Draw grid for reference
    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y), 1)
   
    # Draw obstacles
    for obstacle in obstacles:
        obstacle.draw(screen)
    # Draw car
    car.draw(screen)
    controller.draw(screen)

    # Draw path
    path_planner.draw(screen)
   
    # Display information
    font = pygame.font.SysFont(None, 24)
    speed_text = font.render(f"Speed: {car.speed:.1f}", True, BLACK)
    angle_text = font.render(f"Car Angle: {math.degrees(car.angle):.1f}°", True, BLACK)
   
    steering_text = font.render(f"Steering Angle: {math.degrees(car.steering_angle):.1f}°", True, BLACK)
    collision_text = font.render(f"Collision: {car.collision}", True, RED if car.collision else BLACK)
    mode_text = font.render(f"Mode: {controller.control_mode.upper()}", True, PURPLE if controller.control_mode == "follow" else BLACK)
    trajectory_text = font.render(f"Trajectory points: {len(controller.trajectory)}", True, BLACK)
   
    help_text1 = font.render("R: New obstacles, C: Clear trajectory, T: Toggle mode", True, BLACK)
    help_text2 = font.render("Manual: Arrow keys, Autonomous: Follows trajectory", True, BLACK)
    position_text = font.render(f"Car Position: {car.x:.1f}, {car.y:.1f}", True, BLACK)
   
    screen.blit(speed_text, (10, 10))
    screen.blit(angle_text, (10, 40))
    screen.blit(steering_text, (10, 70))
    screen.blit(collision_text, (10, 100))
    screen.blit(mode_text, (10, 130))
    screen.blit(trajectory_text, (10, 160))
    screen.blit(help_text1, (10, HEIGHT - 50))
    screen.blit(help_text2, (10, HEIGHT - 25))
    screen.blit(position_text, (10, HEIGHT - 75))
   
    # Update display
    pygame.display.flip()
   
    # FPS limit
    clock.tick(60)
pygame.quit()
sys.exit()
