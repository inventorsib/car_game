import pygame
import math
import sys
import random
import numpy as np
from typing import List, Tuple, Optional
import dubins_path_planner

import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 1300, 800
STEP = 2
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Car Obstacle Avoidance with Dubins Path")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (139, 0, 0)

class ArcGenerator:
    """Генератор касательных дуг для объезда препятствий"""
    
    @staticmethod
    def generate_tangent_arc_points(O, R, P, R_arc, num_points=50):
        """
        Генерирует точки дуги от P до T по кратчайшему пути.
        
        Args:
            O: (x, y) центр исходной окружности
            R: радиус исходной окружности
            P: (x, y) начальная точка
            R_arc: радиус дуги скругления
            num_points: количество точек для генерации дуги
            
        Returns:
            list: Список кортежей [(C, T, arc_points)] для каждого решения,
                  где arc_points - список точек от P до T по дуге
        """
        
        def find_tangent_arc_centers(O, R, P, R_arc):
            """Находит центры дуг для внешнего касания"""
            solutions = []
            
            dx = P[0] - O[0]
            dy = P[1] - O[1]
            d = math.sqrt(dx*dx + dy*dy)
            
            if d < (R + R_arc):
                return solutions
            
            a = (d*d + (R + R_arc)**2 - R_arc**2) / (2*d)
            h = math.sqrt((R + R_arc)**2 - a*a)
            
            C0_x = O[0] + a * dx / d
            C0_y = O[1] + a * dy / d
            
            perp_x = -dy * h / d
            perp_y = dx * h / d
            
            C1 = (C0_x + perp_x, C0_y + perp_y)
            C2 = (C0_x - perp_x, C0_y - perp_y)
            
            for C in [C1, C2]:
                vec_OC_x = C[0] - O[0]
                vec_OC_y = C[1] - O[1]
                len_OC = math.sqrt(vec_OC_x**2 + vec_OC_y**2)
                if len_OC == 0:
                    continue
                unit_OC_x = vec_OC_x / len_OC
                unit_OC_y = vec_OC_y / len_OC
                
                T = (O[0] + unit_OC_x * R, O[1] + unit_OC_y * R)
                solutions.append((C, T))
            
            return solutions
        
        def calculate_angle(point, center):
            """Вычисляет угол точки относительно центра"""
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            return math.atan2(dy, dx)
        
        def normalize_angle(angle):
            """Нормализует угол в диапазон [0, 2π]"""
            return angle % (2 * math.pi)
        
        def get_shortest_arc_points(center, start_point, end_point, radius, num_points):
            """Генерирует точки по кратчайшей дуге"""
            start_angle = calculate_angle(start_point, center)
            end_angle = calculate_angle(end_point, center)
            
            # Нормализуем углы
            start_angle_norm = normalize_angle(start_angle)
            end_angle_norm = normalize_angle(end_angle)
            
            # Вычисляем разность углов в правильном направлении
            angle_diff = end_angle_norm - start_angle_norm
            
            # Корректируем для выбора кратчайшего пути
            if angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            elif angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # Генерируем точки дуги
            points = []
            for i in range(num_points + 1):
                t = i / num_points
                current_angle = start_angle + angle_diff * t
                
                x = center[0] + radius * math.cos(current_angle)
                y = center[1] + radius * math.sin(current_angle)
                points.append((x, y))
            
            return points
        
        solutions = []
        arc_centers = find_tangent_arc_centers(O, R, P, R_arc)
        
        for C, T in arc_centers:
            # Генерируем точки дуги от P до T
            arc_points = get_shortest_arc_points(C, P, T, R_arc, num_points)
            solutions.append((C, T, arc_points))
        
        return solutions

class Obstacle:
    def __init__(self, x, y, radius=None):
        self.x = x
        self.y = y
        self.radius = radius if radius else random.randint(25, 45)
        self.safety_radius = self.radius + 40  # Радиус безопасности
        self.color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
       
    def draw(self, surface):
        # Основное препятствие
        pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(surface, BLACK, (int(self.x), int(self.y)), self.radius, 2)
        
        # Зона безопасности
        pygame.draw.circle(surface, (*self.color, 100), (int(self.x), int(self.y)), self.safety_radius, 2)
   
    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                         self.radius * 2, self.radius * 2)
   
    def check_collision(self, car_corners):
        for corner_x, corner_y in car_corners:
            distance = math.sqrt((corner_x - self.x)**2 + (corner_y - self.y)**2)
            if distance <= self.radius:
                return True
        return False

class ObstacleAvoidancePlanner:
    """Упрощенный планировщик объезда препятствий"""
    def __init__(self, nominal_trajectory_y=400):
        self.nominal_trajectory_y = nominal_trajectory_y
        self.safety_margin = 60  # Отступ от препятствий
        
    def find_optimal_avoidance(self, car_pos: Tuple[float, float], 
                         car_angle: float, 
                         obstacles: List[Obstacle]) -> List[Tuple[float, float]]:
        start_x, start_y = car_pos
        
        # Номинальная траектория
        nominal_path = [(x, self.nominal_trajectory_y) for x in np.arange(start_x, WIDTH, STEP)]
        
        # Ищем препятствия на пути
        collision_obstacles = []
        
        for obstacle in obstacles:
            for point in nominal_path:
                dist = math.sqrt((point[0] - obstacle.x)**2 + (point[1] - obstacle.y)**2)
                if dist < obstacle.safety_radius:
                    collision_obstacles.append(obstacle)
                    break
        
        # Если нет препятствий, возвращаем номинальный путь
        if not collision_obstacles:
            return nominal_path
        
        # Берем первое препятствие 
        # TODO: брать ближайшее
        obstacle = collision_obstacles[0]
        
        # Определяем точку входа и выхода из запретной зоны
        entry_angle = math.atan2(self.nominal_trajectory_y - obstacle.y, -obstacle.safety_radius)
        exit_angle = math.atan2(self.nominal_trajectory_y - obstacle.y, obstacle.safety_radius)
        
        sign = 1
        # Определяем сторону объезда
        if self.nominal_trajectory_y < obstacle.y:
            # Движение по нижней дуге
            start_angle = entry_angle
            end_angle = exit_angle
            sign = 1
        else:
            # Движение по верхней дуге
            start_angle = -entry_angle
            end_angle = -exit_angle
            sign = -1

        #! Update by Dubins connections
        # TODO: bruteforce
        #//for x_start in np.arange(obstacle.y - 100, obstacle.y - obstacle.safety_radius, 10):
        #//    for angle_connect in np.arange(0, ) 

        x_start_dub = obstacle.x - 175
        y_start_dub = start_y

        x_circle_dub = obstacle.x
        y_circle_dub = obstacle.y - sign*obstacle.safety_radius

        x_end_dub = obstacle.x + 175
        y_end_dub = start_y

        th_start = 0
        th_circle = 0
        th_end = 0

        curvature = 1/car.max_turning_radius
        path_x_to_zone, path_y_to_zone, _, _, _ = dubins_path_planner.plan_dubins_path(
            x_start_dub, y_start_dub, th_start,
            x_circle_dub, y_circle_dub, th_circle,
            curvature, 0.01
        )

        path_x_from_zone, path_y_from_zone, _, _, _ = dubins_path_planner.plan_dubins_path(
            x_circle_dub, y_circle_dub, th_circle,
            x_end_dub, y_end_dub, th_end,
            curvature, 0.01
        )
       
        # Строим путь
        full_path = []
        
        # 1. Номинальная траектория до точки входа
        entry_x = x_start_dub #//obstacle.x + obstacle.safety_radius * math.cos(start_angle)
        entry_y = y_start_dub  #//obstacle.y+ obstacle.safety_radius * math.sin(start_angle)
        
        for point in nominal_path:
            if point[0] <= entry_x:
                full_path.append(point)
            else:
                break
        
        # 2. Дубинс - выход на запретную зону
        #//arc_points = 100
        #//angle_step = (end_angle - start_angle) / arc_points
        
        for x, y in zip(path_x_to_zone, path_y_to_zone):
            full_path.append((x, y))

        # 3. Движение вдоль запретной зоны

        # 4. Съезд с запретной зоны
        for x, y in zip(path_x_from_zone, path_y_from_zone):
            full_path.append((x, y))
            exit_x = x

        # 5. Номинальная траектория после точки выхода
        #exit_x = obstacle.x + obstacle.safety_radius * math.cos(end_angle)
        
        for point in nominal_path:
            if point[0] >= exit_x:
                full_path.append(point)
        
        return full_path
    
    def draw_debug(self, surface, path):
        """Отрисовка упрощенного пути"""
        if len(path) > 1:
            # Рисуем линии пути
            pygame.draw.lines(surface, GREEN, False, path, 3)
            
            # Рисуем точки пути
            for i, point in enumerate(path):
                color = RED if i == 0 else BLUE if i == len(path) - 1 else PURPLE
                pygame.draw.circle(surface, color, (int(point[0]), int(point[1])), 6)
                
                # Подписи точек
                font = pygame.font.SysFont(None, 20)
                if i == 0:
                    label = "Start"
                elif i == len(path) - 1:
                    label = "End"
                else:
                    label = f"P{i}"
                
                text = font.render(label, True, BLACK)
                surface.blit(text, (point[0] + 8, point[1] - 8))

class Controller:
    def __init__(self):
        self.trajectory = []
        self.target_trajectory = []
        self.max_trajectory_points = 1500
        self.max_target_trajectory_points = 1500
        self.target_point = None
        self.control_mode = "manual"
        self.follow_speed = 0.05
        self.lookahead_distance = 20
        
        # Планировщик объезда
        self.planner = ObstacleAvoidancePlanner()
        self.current_avoidance_path = []
    
    def plan_avoidance(self, car_x, car_y, car_angle, obstacles):
        """Планирует путь объезда"""
        self.current_avoidance_path = self.planner.find_optimal_avoidance(
            (car_x, car_y), car_angle, obstacles
        )
        self.target_trajectory = self.current_avoidance_path.copy()
    
    def add_point(self, x, y):
        self.trajectory.append((x, y))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)
   
    def add_target_point(self, x, y):
        self.target_trajectory.append((x, y))
        if len(self.target_trajectory) > self.max_target_trajectory_points:
            self.target_trajectory.pop(0)

    def draw(self, surface):
        # Отрисовка пути объезда
        if self.current_avoidance_path and len(self.current_avoidance_path) > 1:
            pygame.draw.lines(surface, GREEN, False, self.current_avoidance_path, 3)
            
            # Точки пути
            for point in self.current_avoidance_path[::10]:
                pygame.draw.circle(surface, LIGHT_GREEN, (int(point[0]), int(point[1])), 4)
        
        # Траектория автомобиля
        if len(self.trajectory) > 1:
            for i in range(1, len(self.trajectory)):
                alpha = int(255 * i / len(self.trajectory))
                color = (0, 100, 200)
                pygame.draw.line(surface, color, self.trajectory[i-1], self.trajectory[i], 2)
        
        # Целевая траектория
        if len(self.target_trajectory) > 1:
            for i in range(1, len(self.target_trajectory)):
                alpha = int(255 * i / len(self.target_trajectory))
                color = (100, 0, 200)
                pygame.draw.line(surface, color, self.target_trajectory[i-1], self.target_trajectory[i], 2)
       
        if self.target_point:
            pygame.draw.circle(surface, PURPLE, (int(self.target_point[0]), int(self.target_point[1])), 8)
            pygame.draw.circle(surface, WHITE, (int(self.target_point[0]), int(self.target_point[1])), 8, 2)
   
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
            ahead_idx = min(target_idx + 15, len(self.target_trajectory) - 1)
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

class Car:
    def __init__(self):
        self.x = 100
        self.y = 400
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
        self.collision = False  # Добавлено отсутствующее свойство

        self.max_turning_radius = self.length / math.tan(self.max_steering_angle)
       
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

def generate_obstacles(count=1):
    obstacles = []
    for _ in range(count):
        x = random.randint(400, WIDTH - 200)
        y = random.randint(200, HEIGHT - 200)
        obstacles.append(Obstacle(x, y))
    return obstacles

# Create game objects
car = Car()
obstacles = generate_obstacles(1)
controller = Controller()

# Номинальная траектория
nominal_y = 400
for x in np.arange(0, WIDTH, STEP):
    controller.add_target_point(x, nominal_y)

# Main game loop
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                obstacles = generate_obstacles(1)
                car.collision = False
            elif event.key == pygame.K_c:
                controller.trajectory = []
                controller.target_trajectory = []
                # Восстанавливаем номинальную траекторию
                for x in np.arange(0, WIDTH, STEP):
                    controller.add_target_point(x, nominal_y)
            elif event.key == pygame.K_t:
                controller.control_mode = "follow" if controller.control_mode == "manual" else "manual"
            elif event.key == pygame.K_p:
                # Планирование объезда
                controller.plan_avoidance(car.x, car.y, car.angle, obstacles)
                controller.control_mode = "follow"
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3:  # Right click - set start position
                car.x, car.y = event.pos

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
        # Follow mode
        if abs(car.speed) < 1.0:
            car.speed = 0.65
       
        controller.target_point = controller.find_target_point(car.x, car.y, car.angle)
        if controller.target_point:
            target_steering = controller.calculate_steering(car.x, car.y, car.angle, controller.target_point)
            car.steering_angle = car.steering_angle * 0.7 + target_steering * 0.3

    # Update game state
    car.update()
    if abs(car.speed) > 0.5:
        controller.add_point(car.x, car.y)
    
    # Проверка столкновений
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
        pygame.draw.line(screen, (220, 220, 220), (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, (220, 220, 220), (0, y), (WIDTH, y), 1)
    
    # Номинальная траектория
    pygame.draw.line(screen, GRAY, (0, nominal_y), (WIDTH, nominal_y), 2)
    
    controller.draw(screen)
    car.draw(screen)

    # Draw obstacles
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
    path_text = font.render(f"Avoidance path: {len(controller.current_avoidance_path)} points", True, GREEN if controller.current_avoidance_path else BLACK)
    position_text = font.render(f"Car Position: {car.x:.1f}, {car.y:.1f}", True, BLACK)
    
    help_text1 = font.render("R: New obstacles, C: Clear nominal path, T: Toggle mode", True, BLACK)
    help_text2 = font.render("P: Plan avoidance with tangent arcs, Right click: Set start", True, BLACK)

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