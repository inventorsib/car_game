import numpy as np
import math
import heapq
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

class Node:
    """Узел для Hybrid A*"""
    def __init__(self, x: float, y: float, theta: float, 
                 g: float = 0, h: float = 0, parent=None):
        self.x = x
        self.y = y
        self.theta = theta  # ориентация в радианах
        self.g = g  # стоимость от старта
        self.h = h  # эвристическая стоимость до цели
        self.parent = parent
        
    @property
    def f(self):
        return self.g + self.h
        
    def __lt__(self, other):
        return self.f < other.f

class HybridAStar:
    def __init__(self, grid: np.ndarray, resolution: float = 1.0):
        """
        Args:
            grid: 2D массив препятствий (0 - свободно, 1 - занято)
            resolution: размер ячейки в метрах
        """
        self.grid = grid
        self.resolution = resolution
        self.height, self.width = grid.shape
        
        # Параметры транспортного средства
        self.vehicle_length = 2.5
        self.wheel_base = 1.5
        self.max_steering_angle = math.radians(30)
        
        # Дискретизация углов
        self.angle_resolution = math.radians(10)
        
    def heuristic(self, node: Node, goal: Node) -> float:
        """Эвристическая функция (расстояние + ориентация)"""

        # TODO: heuristic from dubins
        dx = goal.x - node.x
        dy = goal.y - node.y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Учитываем разницу в ориентации
        angle_diff = min(abs(node.theta - goal.theta), 
                        2 * math.pi - abs(node.theta - goal.theta))
        
        return distance + 0.1 * angle_diff
    
    def is_collision(self, x: float, y: float) -> bool:
        """Проверка столкновения"""
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        
        if (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            return self.grid[grid_y, grid_x] == 1
        return True
    
    def get_successors(self, node: Node) -> List[Node]:
        """Генерация преемников с учётом кинематики"""
        successors = []
        steering_angles = [-self.max_steering_angle, 0, self.max_steering_angle]
        step_sizes = [1.0, 2.0]  # разные длины шагов
        
        for steering in steering_angles:
            for step in step_sizes:
                # Кинематическая модель велосипеда
                if abs(steering) < 1e-5:
                    # Движение прямо
                    new_x = node.x + step * math.cos(node.theta)
                    new_y = node.y + step * math.sin(node.theta)
                    new_theta = node.theta
                else:
                    # Поворот
                    turning_radius = self.wheel_base / math.tan(steering)
                    angular_velocity = step / turning_radius
                    
                    new_theta = node.theta + angular_velocity
                    new_x = node.x + turning_radius * (math.sin(new_theta) - math.sin(node.theta))
                    new_y = node.y + turning_radius * (math.cos(node.theta) - math.cos(new_theta))
                
                # Нормализация угла
                new_theta = new_theta % (2 * math.pi)
                
                # Проверка столкновений
                if not self.is_collision(new_x, new_y):
                    cost = step  # базовая стоимость - пройденное расстояние
                    successor = Node(new_x, new_y, new_theta, 
                                   node.g + cost, 0, node)
                    successors.append(successor)
        
        return successors
    
    def discretize_state(self, node: Node) -> Tuple[int, int, int]:
        """Дискретизация состояния для проверки посещённых узлов"""
        x_idx = int(node.x / self.resolution)
        y_idx = int(node.y / self.resolution)
        theta_idx = int(node.theta / self.angle_resolution)
        return (x_idx, y_idx, theta_idx)
    
    def search(self, start: Tuple[float, float, float], 
               goal: Tuple[float, float, float]) -> List[Node]:
        """
        Поиск пути от start до goal
        Args:
            start: (x, y, theta)
            goal: (x, y, theta)
        Returns:
            Список узлов пути или пустой список если путь не найден
        """
        start_node = Node(start[0], start[1], start[2])
        goal_node = Node(goal[0], goal[1], goal[2])
        
        start_node.h = self.heuristic(start_node, goal_node)
        
        open_list = []
        heapq.heappush(open_list, (start_node.f, start_node))
        
        closed_set = set()
        visited = {}  # для отслеживания лучших стоимостей
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            # Проверка достижения цели
            if self.is_goal_reached(current, goal_node):
                return self.reconstruct_path(current)
            
            current_state = self.discretize_state(current)
            if current_state in closed_set:
                continue
                
            closed_set.add(current_state)
            
            for successor in self.get_successors(current):
                successor_state = self.discretize_state(successor)
                
                if successor_state in closed_set:
                    continue
                
                # Обновление эвристики
                successor.h = self.heuristic(successor, goal_node)
                
                # Проверка, не посещали ли мы этот узел с лучшей стоимостью
                if successor_state in visited and visited[successor_state] <= successor.g:
                    continue
                
                visited[successor_state] = successor.g
                heapq.heappush(open_list, (successor.f, successor))
        
        return []  # путь не найден
    
    def is_goal_reached(self, node: Node, goal: Node, pos_tolerance: float = 0.5, 
                       angle_tolerance: float = math.radians(15)) -> bool:
        """Проверка достижения цели"""
        dx = goal.x - node.x
        dy = goal.y - node.y
        distance = math.sqrt(dx**2 + dy**2)
        
        angle_diff = min(abs(node.theta - goal.theta), 
                        2 * math.pi - abs(node.theta - goal.theta))
        
        return distance <= pos_tolerance and angle_diff <= angle_tolerance
    
    def reconstruct_path(self, node: Node) -> List[Node]:
        """Восстановление пути от конечного узла до старта"""
        path = []
        current = node
        while current:
            path.append(current)
            current = current.parent
        return path[::-1]

# Пример использования
def create_test_grid():
    """Создание тестовой карты"""
    grid = np.zeros((50, 50))
    
    # Добавление препятствий
    grid[20:30, 20:30] = 1  # прямоугольное препятствие
    grid[10:15, 10:40] = 1  # стена
    
    return grid

def visualize_path(grid, path, start, goal):
    """Визуализация пути"""
    plt.figure(figsize=(12, 10))
    
    # Отображение карты
    plt.imshow(grid, cmap='Greys', origin='lower')
    
    if path:
        # Отображение пути
        path_x = [node.x for node in path]
        path_y = [node.y for node in path]
        plt.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
        plt.plot(path_x, path_y, 'bo', markersize=3)
    
    # Старт и цель
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    
    # Ориентация
    arrow_length = 2.0
    plt.arrow(start[0], start[1], 
              arrow_length * math.cos(start[2]), arrow_length * math.sin(start[2]),
              head_width=0.5, fc='g', ec='g')
    plt.arrow(goal[0], goal[1],
              arrow_length * math.cos(goal[2]), arrow_length * math.sin(goal[2]),
              head_width=0.5, fc='r', ec='r')
    
    plt.legend()
    plt.grid(True)
    plt.title('Hybrid A* Path Planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Демонстрация
if __name__ == "__main__":
    # Создание карты
    grid = create_test_grid()
    
    # Инициализация планировщика
    planner = HybridAStar(grid, resolution=1.0)
    
    # Задание старта и цели
    start = (5.0, 5.0, math.radians(0))  # (x, y, theta)
    goal = (40.0, 40.0, math.radians(90))
    
    # Поиск пути
    path = planner.search(start, goal)
    
    if path:
        print(f"Путь найден! Длина: {len(path)} узлов")
        print(f"Общая стоимость: {path[-1].g:.2f}")
        
        # Визуализация
        visualize_path(grid, path, start, goal)
    else:
        print("Путь не найден!")