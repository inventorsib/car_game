import pygame
import math
import sys
import random
import numpy as np
from typing import List, Tuple, Optional
import dubins_path_planner
from enum import Enum

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import networkx as nx

from dataclasses import dataclass

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
    _next_id = 1

    def __init__(self, x, y, radius=None):
        self.x = x
        self.y = y
        self.radius = radius if radius else random.randint(25, 45)
        self.safety_radius = self.radius + 40  # Радиус безопасности
        self.color = (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        
        self.id = Obstacle._next_id
        Obstacle._next_id += 1
       
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

# Определим типы узлов
class NodeType(Enum):
    LINE = 1
    CIRCLE = 2
    START = 3
    GOAL = 4

@dataclass
class Node:
    x: float
    y: float
    th: float  # ориентация (theta)
    node_type: NodeType
    circle_id: int = 0
    g_cost: float = 0  # стоимость от старта
    h_cost: float = 0  # эвристическая стоимость до цели
    parent: Optional['Node'] = None
    
    @property
    def f_cost(self):
        return self.g_cost + self.h_cost
    
    def __hash__(self):
        return hash((self.x, self.y, self.th, self.node_type.value, self.circle_id))

class NodeVisualizer:
    def __init__(self, figsize=(10, 8)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.setup_plot()
    
    def setup_plot(self):
        """Настройка графика"""
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X координата')
        self.ax.set_ylabel('Y координата')
        self.ax.set_title('Визуализация узлов пути')
    
    def draw_node(self, node: Node, color='blue', size=100, show_orientation=True):
        """Отрисовка одного узла"""
        # Определяем цвет и маркер в зависимости от типа узла
        if node.node_type == NodeType.START:
            marker = 's'  # квадрат
            color = 'green'
            size = 150
        elif node.node_type == NodeType.GOAL:
            marker = 'D'  # ромб
            color = 'red'
            size = 150
        elif node.node_type == NodeType.CIRCLE:
            marker = 'o'  # круг
            color = 'orange'
        else:  # LINE
            marker = '^'  # треугольник
            color = 'blue'
        
        # Рисуем узел
        self.ax.scatter(node.x, node.y, s=size, c=color, marker=marker, 
                       edgecolors='black', linewidth=1, zorder=3)
        
        # Добавляем подпись с информацией об узле
        label = f"({node.x:.1f}, {node.y:.1f})\nθ={node.th:.1f}"
        if node.node_type == NodeType.CIRCLE:
            label += f"\nID={node.circle_id}"
        
        self.ax.text(node.x, node.y + 0.1, label, fontsize=8, 
                    ha='center', va='bottom', zorder=4)
        
        # Рисуем ориентацию (стрелку направления)
        if show_orientation:
            arrow_length = 0.3
            dx = arrow_length * np.cos(node.th)
            dy = arrow_length * np.sin(node.th)
            self.ax.arrow(node.x, node.y, dx, dy, 
                         head_width=0.05, head_length=0.1, 
                         fc=color, ec='black', zorder=2)
    
    def draw_path(self, nodes, color='purple', linewidth=2, alpha=0.7):
        """Отрисовка пути между узлами"""
        if len(nodes) < 2:
            return
        
        xs = [node.x for node in nodes]
        ys = [node.y for node in nodes]
        
        self.ax.plot(xs, ys, color=color, linewidth=linewidth, 
                    alpha=alpha, zorder=1, linestyle='--')
        
        # Добавляем стрелки для направления пути
        for i in range(len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]
            mid_x = (node1.x + node2.x) / 2
            mid_y = (node1.y + node2.y) / 2
            
            # Вычисляем направление
            dx = node2.x - node1.x
            dy = node2.y - node1.y
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                dx, dy = dx/length, dy/length
                self.ax.arrow(mid_x, mid_y, dx*0.1, dy*0.1,
                            head_width=0.05, head_length=0.1,
                            fc=color, ec=color, zorder=1)
    
    def draw_circle_info(self, node: Node, radius=0.5):
        """Отрисовка дополнительной информации для круговых узлов"""
        if node.node_type == NodeType.CIRCLE:
            # Рисуем дугу окружности
            circle = patches.Arc((node.x, node.y), 
                                width=radius*2, 
                                height=radius*2,
                                angle=0, theta1=0, theta2=np.degrees(node.th),
                                color='orange', linewidth=2, alpha=0.5)
            self.ax.add_patch(circle)
    
    def draw_all_nodes(self, nodes, show_path=True, show_parent_connections=False):
        """Отрисовка всех узлов"""
        # Сортируем узлы по типу для правильного порядка отрисовки
        start_nodes = [n for n in nodes if n.node_type == NodeType.START]
        goal_nodes = [n for n in nodes if n.node_type == NodeType.GOAL]
        line_nodes = [n for n in nodes if n.node_type == NodeType.LINE]
        circle_nodes = [n for n in nodes if n.node_type == NodeType.CIRCLE]
        
        # Отрисовываем в определенном порядке
        for node in line_nodes:
            self.draw_node(node)
        
        for node in circle_nodes:
            self.draw_node(node)
            self.draw_circle_info(node)
        
        for node in start_nodes:
            self.draw_node(node)
        
        for node in goal_nodes:
            self.draw_node(node)
        
        # Отрисовываем связи с родительскими узлами
        if show_parent_connections:
            for node in nodes:
                if node.parent:
                    self.ax.plot([node.x, node.parent.x], 
                                [node.y, node.parent.y], 
                                'gray', linestyle=':', alpha=0.5, zorder=1)
        
        # Отрисовываем путь, если узлы упорядочены
        if show_path and len(nodes) > 1:
            # Пытаемся восстановить путь через parent ссылки
            path_nodes = []
            current = nodes[-1] if nodes[-1].node_type == NodeType.GOAL else None
            
            while current:
                path_nodes.append(current)
                current = current.parent
            
            if path_nodes:
                path_nodes.reverse()
                self.draw_path(path_nodes)
    
    def show_legend(self):
        """Добавление легенды"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                  markersize=10, label='Старт'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='red',
                  markersize=10, label='Цель'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue',
                  markersize=10, label='Линейный узел'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                  markersize=10, label='Круговой узел'),
            Line2D([0], [0], color='purple', linestyle='--', linewidth=2,
                  label='Путь'),
            Line2D([0], [0], color='gray', linestyle=':', linewidth=1,
                  label='Родительские связи'),
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right')
    
    def show(self):
        """Показать график"""
        self.show_legend()
        plt.tight_layout()
        plt.show()

class GraphVisualizer:
    def __init__(self, figsize=(12, 10), max_connection_dist=200, 
                 max_curvature=0.02, step_size=1.0):
        """
        Инициализация визуализатора графа с путями Дубинса
        
        Args:
            figsize: Размер графика
            max_connection_dist: Максимальное расстояние для соединения узлов
            max_curvature: Максимальная кривизна (1/минимальный радиус поворота)
            step_size: Шаг дискретизации для построения пути
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.graph = nx.Graph()
        self.max_connection_dist = max_connection_dist
        self.max_curvature = max_curvature
        self.step_size = step_size
        self.dubins_cache = {}  # Кэш для путей Дубинса
        self.setup_plot()
    
    def setup_plot(self):
        """Настройка графика"""
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X координата', fontsize=12)
        self.ax.set_ylabel('Y координата', fontsize=12)
        self.ax.set_title('Граф с путями Дубинса', fontsize=14)
    
    def is_connection_possible(self, node1: Node, node2: Node, 
                               obstacles: List[Obstacle]) -> bool:
        """
        Проверяет, возможен ли переход между узлами с помощью пути Дубинса
        
        Args:
            node1: Начальный узел
            node2: Конечный узел
            obstacles: Список препятствий
            
        Returns:
            bool: True если соединение возможно
        """
        # Узел должен быть правее (больше по X)
        if node2.x <= node1.x:
            return False
        
        # Максимальное расстояние для соединения
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        euclidean_dist = math.sqrt(dx*dx + dy*dy)
        
        if euclidean_dist > self.max_connection_dist:
            return False
        
        # Генерируем путь Дубинса для проверки столкновений
        path_points, path_length = self._generate_and_cache_dubins_path(node1, node2)
        
        if path_points is None or len(path_points) == 0:
            return False
        
        # Проверка столкновений с препятствиями
        for obstacle in obstacles:
            if self._dubins_path_collision(path_points, obstacle):
                return False
        
        return True
    
    def _generate_and_cache_dubins_path(self, node1: Node, node2: Node, num_points=100):
        """
        Генерирует путь Дубинса и кэширует результат
        
        Args:
            node1, node2: начальный и конечный узлы
            num_points: количество точек для дискретизации
            
        Returns:
            tuple: (точки пути, длина пути)
        """
        # Создаем ключ для кэша
        cache_key = (node1.x, node1.y, node1.th, 
                    node2.x, node2.y, node2.th,
                    self.max_curvature)
        
        # Проверяем кэш
        if cache_key in self.dubins_cache:
            return self.dubins_cache[cache_key]
        
        # Генерируем путь Дубинса с использованием вашего модуля
        try:
            # Используем функцию plan_dubins_path из вашего модуля
            x_list, y_list, yaw_list, modes, lengths = dubins.plan_dubins_path(
                s_x=node1.x,
                s_y=node1.y,
                s_yaw=node1.th,
                g_x=node2.x,
                g_y=node2.y,
                g_yaw=node2.th,
                curvature=self.max_curvature,
                step_size=self.step_size
            )
            
            # Преобразуем в массив точек
            path_points = np.column_stack([x_list, y_list])
            
            # Вычисляем общую длину пути
            total_length = sum(lengths)
            
            # Сохраняем в кэш
            self.dubins_cache[cache_key] = (path_points, total_length)
            
            return path_points, total_length
            
        except Exception as e:
            print(f"Ошибка при генерации пути Дубинса: {e}")
            # Возвращаем упрощенный путь в случае ошибки
            points = self._generate_simple_path(node1, node2, num_points)
            distance = math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
            return points, distance
    
    def _generate_simple_path(self, node1: Node, node2: Node, num_points=50):
        """
        Генерирует упрощенный путь (используется как fallback)
        """
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            x = node1.x + t * (node2.x - node1.x)
            y = node1.y + t * (node2.y - node1.y)
            points.append([x, y])
        
        return np.array(points)
    
    def _dubins_path_collision(self, path_points: np.ndarray, obstacle: Obstacle) -> bool:
        """
        Проверяет столкновение пути Дубинса с препятствием
        
        Args:
            path_points: точки пути [N, 2]
            obstacle: препятствие
            
        Returns:
            bool: True если есть столкновение
        """
        # Проверяем каждую точку пути
        for point in path_points:
            distance = math.sqrt((point[0] - obstacle.x)**2 + (point[1] - obstacle.y)**2)
            if distance <= obstacle.safety_radius:
                return True
        
        # Также проверяем промежуточные сегменты пути
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i + 1]
            
            if self._line_circle_collision(p1, p2, obstacle):
                return True
        
        return False
    
    def _line_circle_collision(self, p1, p2, obstacle: Obstacle) -> bool:
        """
        Проверка пересечения линии с окружностью
        """
        # Вектор линии
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        line_len = math.sqrt(dx*dx + dy*dy)
        
        if line_len == 0:
            return True
        
        # Вектор от начала линии к центру окружности
        fx = obstacle.x - p1[0]
        fy = obstacle.y - p1[1]
        
        # Проекция на линию
        projection = (fx*dx + fy*dy) / line_len
        
        # Ближайшая точка на линии к центру окружности
        if projection <= 0:
            closest_x, closest_y = p1[0], p1[1]
        elif projection >= line_len:
            closest_x, closest_y = p2[0], p2[1]
        else:
            closest_x = p1[0] + (dx * projection / line_len)
            closest_y = p1[1] + (dy * projection / line_len)
        
        # Расстояние от ближайшей точки до центра окружности
        dist = math.sqrt((obstacle.x - closest_x)**2 + (obstacle.y - closest_y)**2)
        
        return dist <= obstacle.safety_radius
    
    def calculate_dubins_distance(self, node1: Node, node2: Node) -> float:
        """
        Вычисление длины пути Дубинса между двумя узлами
        
        Args:
            node1, node2: начальный и конечный узлы
            
        Returns:
            float: длина пути Дубинса
        """
        # Генерируем путь и получаем его длину
        _, path_length = self._generate_and_cache_dubins_path(node1, node2)
        return path_length
    
    def build_graph(self, nodes: List[Node], obstacles: List[Obstacle]) -> nx.Graph:
        """
        Построение графа с использованием путей Дубинса
        
        Args:
            nodes: Список всех узлов
            obstacles: Список препятствий
            
        Returns:
            nx.Graph: Построенный граф
        """
        self.graph.clear()
        self.dubins_cache.clear()  # Очищаем кэш
        
        # Добавляем узлы в граф с атрибутами
        for i, node in enumerate(nodes):
            self.graph.add_node(i, 
                               x=node.x, 
                               y=node.y, 
                               th=node.th,
                               node_type=node.node_type,
                               circle_id=node.circle_id,
                               g_cost=node.g_cost,
                               h_cost=node.h_cost)
        
        # Проверяем все возможные соединения
        print("Построение графа с путями Дубинса...")
        total_checks = 0
        connections_made = 0
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                total_checks += 1
                
                if self.is_connection_possible(nodes[i], nodes[j], obstacles):
                    # Вычисляем длину пути Дубинса
                    dubins_distance = self.calculate_dubins_distance(nodes[i], nodes[j])
                    
                    # Евклидово расстояние для сравнения
                    euclidean_distance = math.sqrt(
                        (nodes[j].x - nodes[i].x)**2 + 
                        (nodes[j].y - nodes[i].y)**2)
                    
                    # Учитываем тип узлов в стоимости
                    cost_multiplier = 1.0
                    if (nodes[i].node_type == NodeType.CIRCLE or 
                        nodes[j].node_type == NodeType.CIRCLE):
                        cost_multiplier = 1.2  # Дополнительная стоимость для круговых узлов
                    
                    # Сохраняем информацию о ребре
                    self.graph.add_edge(i, j, 
                                       weight=dubins_distance * cost_multiplier,
                                       dubins_distance=dubins_distance,
                                       euclidean_distance=euclidean_distance,
                                       start_node=i,
                                       end_node=j)
                    connections_made += 1
        
        print(f"Проверено соединений: {total_checks}")
        print(f"Установлено соединений: {connections_made}")
        print(f"Плотность графа: {connections_made/total_checks*100:.1f}%")
        
        # Статистика по длинам путей
        if connections_made > 0:
            dubins_lengths = [d['dubins_distance'] for _, _, d in self.graph.edges(data=True)]
            euclid_lengths = [d['euclidean_distance'] for _, _, d in self.graph.edges(data=True)]
            
            avg_ratio = sum(dubins_lengths) / sum(euclid_lengths) if sum(euclid_lengths) > 0 else 1.0
            print(f"Среднее отношение Дубинс/Евклид: {avg_ratio:.3f}")
        
        return self.graph
    
    def draw_obstacles(self, obstacles: List[Obstacle]):
        """Отрисовка препятствий"""
        for obstacle in obstacles:
            # Основной радиус
            circle_color = [c/255 for c in obstacle.color]
            circle = patches.Circle((obstacle.x, obstacle.y), obstacle.radius,
                                  color=circle_color, 
                                  alpha=0.6, 
                                  label=f'Препятствие {obstacle.id}' if obstacle.id == 1 else "")
            self.ax.add_patch(circle)
            
            # Радиус безопасности
            safety_circle = patches.Circle((obstacle.x, obstacle.y), obstacle.safety_radius,
                                         color=circle_color, 
                                         alpha=0.2, 
                                         linestyle='--', 
                                         fill=False,
                                         label='Радиус безопасности' if obstacle.id == 1 else "")
            self.ax.add_patch(safety_circle)
            
            # Подпись с ID
            self.ax.text(obstacle.x, obstacle.y, f"ID:{obstacle.id}", 
                        ha='center', va='center', 
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor="white", alpha=0.8))
    
    def draw_nodes(self, nodes: List[Node], highlight_path: List[int] = None):
        """Отрисовка узлов графа"""
        if highlight_path is None:
            highlight_path = []
        
        # Создаем словарь позиций
        pos = {}
        for i, node in enumerate(nodes):
            pos[i] = (node.x, node.y)
        
        # Подготавливаем данные для отрисовки
        node_colors = []
        node_sizes = []
        
        for i, node in enumerate(nodes):
            if i in highlight_path:
                node_colors.append('gold')
                node_sizes.append(350)
            elif node.node_type == NodeType.START:
                node_colors.append('green')
                node_sizes.append(300)
            elif node.node_type == NodeType.GOAL:
                node_colors.append('red')
                node_sizes.append(300)
            elif node.node_type == NodeType.CIRCLE:
                node_colors.append('orange')
                node_sizes.append(250)
            else:  # LINE
                node_colors.append('blue')
                node_sizes.append(200)
        
        # Рисуем узлы
        nx.draw_networkx_nodes(self.graph, pos, 
                             node_color=node_colors,
                             node_size=node_sizes,
                             edgecolors='black',
                             linewidths=1.5,
                             ax=self.ax)
        
        # Добавляем подписи к узлам
        for i, (x, y) in pos.items():
            node = nodes[i]
            label = f"N{i}"
            
            if node.node_type == NodeType.CIRCLE:
                label += f"\nC{node.circle_id}"
            
            self.ax.text(x, y + 0.2, label, 
                        ha='center', va='bottom', 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", 
                                 facecolor="white", alpha=0.8))
            
            # Рисуем ориентацию (стрелку)
            arrow_length = 0.4
            dx = arrow_length * math.cos(node.th)
            dy = arrow_length * math.sin(node.th)
            
            self.ax.arrow(x, y, dx, dy, 
                         head_width=0.08, head_length=0.15, 
                         fc=node_colors[i], ec='black', 
                         alpha=0.8, linewidth=1.5)
    
    def draw_dubins_edges(self, nodes: List[Node], highlight_path: List[int] = None):
        """Отрисовка рёбер графа как путей Дубинса"""
        if highlight_path is None:
            highlight_path = []
        
        # Отрисовываем все пути Дубинса
        for u, v, data in self.graph.edges(data=True):
            node1 = nodes[u]
            node2 = nodes[v]
            
            # Определяем стиль отрисовки
            if u in highlight_path and v in highlight_path:
                # Ребро пути - выделяем
                color = 'red'
                linewidth = 3
                alpha = 0.9
                zorder = 3
                label = 'Оптимальный путь' if u == highlight_path[0] else None
            else:
                # Обычное ребро
                color = 'gray'
                linewidth = 1
                alpha = 0.3
                zorder = 1
                label = None
            
            # Получаем кэшированный путь Дубинса
            cache_key = (node1.x, node1.y, node1.th, 
                        node2.x, node2.y, node2.th,
                        self.max_curvature)
            
            if cache_key in self.dubins_cache:
                path_points, _ = self.dubins_cache[cache_key]
                
                # Рисуем путь
                self.ax.plot(path_points[:, 0], path_points[:, 1], 
                            color=color, linewidth=linewidth, 
                            alpha=alpha, zorder=zorder, label=label)
                
                # Добавляем стрелку направления в середине пути
                if len(path_points) > 2:
                    mid_idx = len(path_points) // 2
                    if mid_idx + 1 < len(path_points):
                        dx = path_points[mid_idx + 1, 0] - path_points[mid_idx, 0]
                        dy = path_points[mid_idx + 1, 1] - path_points[mid_idx, 1]
                        length = math.sqrt(dx*dx + dy*dy)
                        if length > 0:
                            self.ax.arrow(path_points[mid_idx, 0], 
                                         path_points[mid_idx, 1],
                                         dx*0.3, dy*0.3,
                                         head_width=0.05, head_length=0.1,
                                         fc=color, ec=color, alpha=alpha*0.8, 
                                         zorder=zorder)
    
    def find_shortest_path(self, start_idx: int, goal_idx: int) -> List[int]:
        """
        Найти кратчайший путь в графе
        
        Args:
            start_idx: Индекс начального узла
            goal_idx: Индекс конечного узла
            
        Returns:
            List[int]: Список индексов узлов пути или пустой список
        """
        try:
            if not self.graph.has_node(start_idx) or not self.graph.has_node(goal_idx):
                print(f"Ошибка: узел {start_idx} или {goal_idx} не существует в графе")
                return []
            
            # Используем алгоритм Дейкстры с весами пути Дубинса
            path = nx.dijkstra_path(self.graph, start_idx, goal_idx, weight='weight')
            path_length = nx.dijkstra_path_length(self.graph, start_idx, goal_idx, weight='weight')
            
            # Вычисляем сумму евклидовых расстояний для сравнения
            euclid_length = 0
            for i in range(len(path) - 1):
                euclid_length += self.graph.edges[path[i], path[i+1]]['euclidean_distance']
            
            print(f"✓ Найден путь из {len(path)} узлов")
            print(f"✓ Длина пути (Дубинс): {path_length:.2f}")
            print(f"✓ Длина пути (Евклид): {euclid_length:.2f}")
            print(f"✓ Отношение: {path_length/euclid_length:.3f}")
            print(f"✓ Путь: {' → '.join(map(str, path))}")
            
            return path
            
        except nx.NetworkXNoPath:
            print("✗ Путь не найден!")
            return []
        except Exception as e:
            print(f"✗ Ошибка при поиске пути: {e}")
            return []
    
    def draw_graph_statistics(self, nodes: List[Node]):
        """Отображение статистики графа"""
        if self.graph.number_of_nodes() == 0:
            return
        
        stats_text = [
            f"СТАТИСТИКА ГРАФА:",
            f"Узлы: {self.graph.number_of_nodes()}",
            f"Соединения: {self.graph.number_of_edges()}",
            f"Плотность: {nx.density(self.graph):.3f}",
        ]
        
        # Статистика по типам узлов
        type_counts = {}
        for i in self.graph.nodes():
            node_type = self.graph.nodes[i]['node_type']
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        stats_text.append(f"Типы узлов:")
        for node_type, count in type_counts.items():
            stats_text.append(f"  {node_type.name}: {count}")
        
        # Статистика по длинам путей
        if self.graph.number_of_edges() > 0:
            dubins_lengths = [d['dubins_distance'] for _, _, d in self.graph.edges(data=True)]
            euclid_lengths = [d['euclidean_distance'] for _, _, d in self.graph.edges(data=True)]
            
            avg_dubins = sum(dubins_lengths) / len(dubins_lengths)
            avg_euclid = sum(euclid_lengths) / len(euclid_lengths)
            avg_ratio = avg_dubins / avg_euclid if avg_euclid > 0 else 1.0
            
            stats_text.extend([
                f"Средняя длина:",
                f"  Дубинс: {avg_dubins:.1f}",
                f"  Евклид: {avg_euclid:.1f}",
                f"  Отношение: {avg_ratio:.3f}",
            ])
        
        # Параметры планирования
        stats_text.extend([
            f"Параметры:",
            f"  Макс. соединение: {self.max_connection_dist}",
            f"  Кривизна: {self.max_curvature:.4f}",
            f"  Мин. радиус: {1.0/self.max_curvature if self.max_curvature > 0 else '∞':.1f}",
            f"  Шаг: {self.step_size}",
        ])
        
        # Создаем текстовое поле со статистикой
        stats_str = "\n".join(stats_text)
        self.ax.text(0.02, 0.98, stats_str,
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.5",
                             facecolor="lightyellow",
                             edgecolor="orange",
                             alpha=0.9))
    
    def show_legend(self):
        """Добавление легенды"""
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
                  markersize=12, label='Старт', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                  markersize=12, label='Цель', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                  markersize=10, label='Круговой узел', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                  markersize=10, label='Линейный узел', markeredgewidth=1.5),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold',
                  markersize=12, label='Узел пути', markeredgewidth=1.5),
            Line2D([0], [0], color='red', linewidth=3, label='Оптимальный путь'),
            Line2D([0], [0], color='gray', linewidth=1, label='Возможные пути'),
            Patch(facecolor='lightgray', alpha=0.6, label='Препятствие'),
            Patch(facecolor='lightgray', alpha=0.2, edgecolor='gray', 
                  linestyle='--', label='Радиус безопасности'),
        ]
        
        self.ax.legend(handles=legend_elements, 
                      loc='upper left', 
                      bbox_to_anchor=(1.02, 1),
                      borderaxespad=0.,
                      fontsize=10,
                      title="Легенда",
                      title_fontsize=11)
    
    def visualize(self, nodes: List[Node], obstacles: List[Obstacle],
                 auto_find_path: bool = True) -> List[int]:
        """
        Полная визуализация графа с путями Дубинса
        
        Args:
            nodes: Список узлов
            obstacles: Список препятствий
            auto_find_path: Автоматически искать путь
            
        Returns:
            List[int]: Найденный путь или пустой список
        """
        # Строим граф
        self.build_graph(nodes, obstacles)
        
        # Очищаем график
        self.ax.clear()
        self.setup_plot()
        
        # Находим индексы старта и цели
        start_idx = next((i for i, n in enumerate(nodes) 
                         if n.node_type == NodeType.START), 0)
        goal_idx = next((i for i, n in enumerate(nodes) 
                        if n.node_type == NodeType.GOAL), len(nodes)-1)
        
        # Ищем путь
        path = []
        if auto_find_path and start_idx is not None and goal_idx is not None:
            path = self.find_shortest_path(start_idx, goal_idx)
        
        # Отрисовываем препятствия
        self.draw_obstacles(obstacles)
        
        # Отрисовываем пути Дубинса
        self.draw_dubins_edges(nodes, path)
        
        # Отрисовываем узлы
        self.draw_nodes(nodes, path)
        
        # Добавляем статистику
        self.draw_graph_statistics(nodes)
        
        # Добавляем легенду
        self.show_legend()
        
        # Настраиваем layout
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # Автомасштабирование
        self._auto_scale(nodes, obstacles)
        
        # Выводим информацию о пути
        if path:
            self._print_path_details(path, nodes)
        
        return path
    
    def _auto_scale(self, nodes: List[Node], obstacles: List[Obstacle]):
        """Автомасштабирование графика"""
        all_x = [node.x for node in nodes]
        all_y = [node.y for node in nodes]
        
        if obstacles:
            all_x.extend([obstacle.x for obstacle in obstacles])
            all_y.extend([obstacle.y for obstacle in obstacles])
        
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            
            x_margin = max((x_max - x_min) * 0.15, 50)
            y_margin = max((y_max - y_min) * 0.15, 50)
            
            self.ax.set_xlim(x_min - x_margin, x_max + x_margin)
            self.ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    def _print_path_details(self, path: List[int], nodes: List[Node]):
        """Вывод детальной информации о пути"""
        print("\n" + "="*60)
        print("ДЕТАЛЬНАЯ ИНФОРМАЦИЯ О ПУТИ ДУБИНСА:")
        print("="*60)
        
        total_dubins_length = 0
        total_euclid_length = 0
        
        for i, node_idx in enumerate(path):
            node = nodes[node_idx]
            
            print(f"{i+1:2d}. Узел {node_idx:2d}: "
                  f"({node.x:6.1f}, {node.y:6.1f}) | "
                  f"θ={node.th:5.2f} | "
                  f"Тип: {node.node_type.name:8s}")
            
            if i < len(path) - 1:
                next_idx = path[i+1]
                edge_data = self.graph.edges[node_idx, next_idx]
                
                dubins_dist = edge_data['dubins_distance']
                euclid_dist = edge_data['euclidean_distance']
                
                total_dubins_length += dubins_dist
                total_euclid_length += euclid_dist
                
                print(f"     → Дубинс: {dubins_dist:6.1f} | "
                      f"Евклид: {euclid_dist:6.1f} | "
                      f"Коэф: {dubins_dist/euclid_dist:.3f}")
        
        print(f"\n📊 ИТОГО: {len(path)} сегментов")
        print(f"📊 Общая длина Дубинса: {total_dubins_length:.1f}")
        print(f"📊 Общая длина Евклида: {total_euclid_length:.1f}")
        print(f"📊 Средний коэффициент: {total_dubins_length/total_euclid_length:.3f}")
        print("="*60)

class ObstacleAvoidancePlanner:
    """Упрощенный планировщик объезда препятствий"""
    def __init__(self, car, nominal_trajectory_y=400, step=20):
        self.nominal_trajectory_y = nominal_trajectory_y
        self.safety_margin = 60  # Отступ от препятствий
        self.curvature = 1/car.max_turning_radius
        self.graph = nx.Graph()
        self.car = car
        self.step = step

    def crate_nodes(self, obstacles):
        nodes = []
        
        # Создаем узлы вдоль номинальной траектории
        for x in np.arange(0, WIDTH, self.step):
            y = self.nominal_trajectory_y
            node_type = NodeType.LINE
            th = 0.0  # Направление вдоль оси X
            circle_id = -1  # -1 означает "не на окружности"
            
            # Проверяем столкновение с препятствиями
            for obstacle in obstacles:
                # Расстояние от точки до центра препятствия
                dx = x - obstacle.x
                dy = y - obstacle.y
                distance_sq = dx**2 + dy**2
                
                if distance_sq < obstacle.safety_radius**2:

                    for sign in (-1, 1):
                        y = obstacle.y + sign * np.sqrt(obstacle.safety_radius**2 - (x - obstacle.x)**2)
                        dy_dx = - (x - obstacle.x) / (y - obstacle.y)  # поскольку знак уже учтён в y
                        th = np.arctan2(dy_dx, 1)
                        node_type = NodeType.CIRCLE
                        circle_id = obstacle.id

                        node = Node(x, y, th, node_type, circle_id)
                        nodes.append(node)


            node = Node(x, y, th, node_type, circle_id)
            nodes.append(node)
            
        # Добавляем конечный узел
        end_node = Node(WIDTH - 50, self.nominal_trajectory_y, 0, NodeType.LINE, -1)
        nodes.append(end_node)
        
        self.debug_nodes = nodes.copy()
        return nodes
    
    def _line_circle_collision(self, p1n, p2n, obstacle: Obstacle) -> bool:
        """
        Проверка пересечения линии с окружностью
        """
        p1 = (p1n.x, p1n.y)
        p2 = (p2n.x, p2n.y)

        # Вектор линии
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        line_len = math.sqrt(dx*dx + dy*dy)
        
        if line_len == 0:
            return True
        
        # Вектор от начала линии к центру окружности
        fx = obstacle.x - p1[0]
        fy = obstacle.y - p1[1]
        
        # Проекция на линию
        projection = (fx*dx + fy*dy) / line_len
        
        # Ближайшая точка на линии к центру окружности
        if projection <= 0:
            closest_x, closest_y = p1[0], p1[1]
        elif projection >= line_len:
            closest_x, closest_y = p2[0], p2[1]
        else:
            closest_x = p1[0] + (dx * projection / line_len)
            closest_y = p1[1] + (dy * projection / line_len)
        
        # Расстояние от ближайшей точки до центра окружности
        dist = math.sqrt((obstacle.x - closest_x)**2 + (obstacle.y - closest_y)**2)
        
        return dist <= obstacle.safety_radius*0.8

    def is_connection_possible(self, node1: Node, node2: Node, 
                              obstacles: List[Obstacle]) -> bool:
        """Проверяет, возможен ли переход между узлами"""
        
        # Узел должен быть правее (больше по X)
        if node2.x <= node1.x:
            return False
        
        # Максимальное расстояние для соединения
        max_connection_dist = 1500
        dist = math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
        if dist > max_connection_dist:
            return False
        
        # Проверка столкновений с препятствиями
        for obstacle in obstacles:
            if self._line_circle_collision(node1, node2, obstacle):
                return False
            
        if (node1.node_type == node2.node_type == NodeType.CIRCLE) and (node1.circle_id==node2.circle_id):
            dist = math.sqrt((node2.x - node1.x)**2 + (node2.y - node1.y)**2)
            if dist > self.step*1.5:
                return False
        
        # Проверка минимального радиуса поворота
        # if node1.node_type == NodeType.CIRCLE or node2.node_type == NodeType.CIRCLE:
        #     # Для движений с поворотом проверяем минимальный радиус
        #     angle_diff = abs(node2.th - node1.th)
        #     if angle_diff > 0.1:  # Есть значительный поворот
        #         # Упрощенная проверка: требуемый радиус не должен быть меньше максимального
        #         if dist / (2 * math.sin(angle_diff/2)) < self.car.max_steering_angle * 0.8:
        #             return False
        
        return True


    def build_graph(self, nodes: List[Node], obstacles: List[Obstacle]) -> nx.Graph:

        self.graph.clear()
        
        # Добавляем узлы в граф с атрибутами
        for i, node in enumerate(nodes):
            self.graph.add_node(i, 
                               x=node.x, 
                               y=node.y, 
                               th=node.th,
                               node_type=node.node_type,
                               circle_id=node.circle_id,
                               g_cost=node.g_cost,
                               h_cost=node.h_cost)
        
        # Проверяем все возможные соединения
        print("Построение графа с путями Дубинса")
        total_checks = 0
        connections_made = 0

        #plt.axis('equal')
        #plt.grid(True)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                total_checks += 1

                first_node = nodes[i]
                second_node = nodes[j]

                # if first_node.node_type == NodeType.CIRCLE:
                #     #plt.plot(first_node.x, first_node.y, '+', markersize=15)      
                    
                # if second_node.node_type == NodeType.CIRCLE:
                #     #plt.plot(second_node.x, second_node.y, '*', markersize=15)  
                
                if self.is_connection_possible(nodes[i], nodes[j], obstacles):
                    # Вычисляем длину пути Дубинса

                    # переход с линии на линию
                    if first_node.node_type == NodeType.LINE and second_node.node_type == NodeType.LINE:
                        path_x = np.linspace(first_node.x, second_node.x, 5)
                        path_y = np.linspace(first_node.y, second_node.y, 5) 
                        total_length = np.sqrt(np.square(first_node.x-second_node.x)+np.square(first_node.y-second_node.y))   
                    # переход с линии на окружность или наоборот
                    elif first_node.node_type != second_node.node_type:
                        path_x, path_y, yaws, modes, lengths  = dubins_path_planner.plan_dubins_path(
                            first_node.x, first_node.y, first_node.th,
                            second_node.x, second_node.y, second_node.th,
                            self.curvature, 0.05
                        )  
                        total_length = sum(lengths)
                    # переход с окружности на другую окружность
                    elif (first_node.node_type == second_node.node_type == NodeType.CIRCLE) and (first_node.circle_id!=second_node.circle_id):
                        path_x, path_y, yaws, modes, lengths  = dubins_path_planner.plan_dubins_path(
                            first_node.x, first_node.y, first_node.th,
                            second_node.x, second_node.y, second_node.th,
                            self.curvature, 0.05
                        )
                    # переход с окружности на эту же окружность (TODO: now by LINE)                
                    else:
                        path_x = np.linspace(first_node.x, second_node.x, 5), 
                        path_y =  np.linspace(first_node.y, second_node.y, 5), 
                        total_length = np.sqrt(np.square(first_node.x-second_node.x)+np.square(first_node.y-second_node.y))   

                    dubins_distance = total_length
                    
                    #plt.plot(path_x, path_y, linewidth=1)
                    #plt.plot(first_node.x, first_node.y, 'o')
                    #plt.plot(second_node.x, second_node.y, '*')
                    # Учитываем тип узлов в стоимости
                    cost_multiplier = 1.0
                    if (nodes[i].node_type == NodeType.CIRCLE or 
                        nodes[j].node_type == NodeType.CIRCLE):
                        cost_multiplier = 1.5  # Дополнительная стоимость для круговых узлов
                    
                    # Сохраняем информацию о ребре
                    self.graph.add_edge(i, j, 
                                       weight=dubins_distance*cost_multiplier,
                                       dubins_distance=dubins_distance,
                                       start_node=i,
                                       end_node=j)
                    connections_made += 1
        #plt.show()
        print(f"Проверено соединений: {total_checks}")
        print(f"Установлено соединений: {connections_made}")
        print(f"Плотность графа: {connections_made/total_checks*100:.1f}%")

        
        return self.graph
        
    
    def find_optimal_avoidance(self, car_pos: Tuple[float, float], 
                         car_angle: float, 
                         obstacles: List[Obstacle]) -> List[Tuple[float, float]]:
        nodes = self.crate_nodes(obstacles)
        self.build_graph(nodes, obstacles)

        start_node = 0
        end_node = len(nodes)-2
        path_nodes = nx.dijkstra_path(self.graph, start_node, end_node, weight='weight')

        trajectory_points = []
        
        for first_node_indx, second_node_indx in zip(path_nodes[:-1], path_nodes[1:]):
            first_node = nodes[first_node_indx]
            second_node = nodes[second_node_indx]
            
            # переход с линии на линию
            if first_node.node_type == NodeType.LINE and second_node.node_type == NodeType.LINE:
                path_x = np.linspace(first_node.x, second_node.x, 150)
                path_y = np.linspace(first_node.y, second_node.y, 150)
                total_length = np.sqrt(np.square(first_node.x-second_node.x)+np.square(first_node.y-second_node.y))
            
            # переход с линии на окружность или наоборот
            elif first_node.node_type != second_node.node_type:
                path_x, path_y, yaws, modes, lengths = dubins_path_planner.plan_dubins_path(
                    first_node.x, first_node.y, first_node.th,
                    second_node.x, second_node.y, second_node.th,
                    self.curvature, 0.05
                )
            
            # переход с окружности на другую окружность
            elif (first_node.node_type == second_node.node_type == NodeType.CIRCLE) and (first_node.circle_id != second_node.circle_id):
                path_x, path_y, yaws, modes, lengths = dubins_path_planner.plan_dubins_path(
                    first_node.x, first_node.y, first_node.th,
                    second_node.x, second_node.y, second_node.th,
                    self.curvature, 0.05
                )
            
            # переход с окружности на эту же окружность
            else:
                path_x = np.linspace(first_node.x, second_node.x, 10)
                path_y = np.linspace(first_node.y, second_node.y, 10)
                total_length = np.sqrt(np.square(first_node.x-second_node.x)+np.square(first_node.y-second_node.y))
            
            # Добавляем точки в список, исключая возможные дубликаты в стыках сегментов
            for x, y in zip(path_x, path_y):
                # Пропускаем первую точку сегмента (кроме первого сегмента), 
                # так как она совпадает с последней точкой предыдущего сегмента
                if len(trajectory_points) > 0 and np.isclose(x, trajectory_points[-1][0]) and np.isclose(y, trajectory_points[-1][1]):
                    continue
                trajectory_points.append((float(x), float(y)))
        
        return trajectory_points
        
    
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
    def __init__(self, car):
        self.trajectory = []
        self.target_trajectory = []
        self.max_trajectory_points = 1500
        self.max_target_trajectory_points = 1500
        self.target_point = None
        self.control_mode = "manual"
        self.follow_speed = 0.05
        self.lookahead_distance = 20
        
        # Планировщик объезда
        self.planner = ObstacleAvoidancePlanner(car=car)
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

    #for _ in range(count):
    x = random.randint(-10, 10) + 500
    y = random.randint(-10, 10) + 400
    obstacles.append(Obstacle(x, y))
    
    x = random.randint(-200, 200) + 800
    y = random.randint(-10, 10) + 400
    obstacles.append(Obstacle(x, y))
    
    x = random.randint(-200, 200) + 600
    y = random.randint(-200, 200) + 400
    obstacles.append(Obstacle(x, y))
    
    x = random.randint(-200, 200) + 600
    y = random.randint(-200, 200) + 400
    obstacles.append(Obstacle(x, y))

    return obstacles

# Create game objects
car = Car()
obstacles = generate_obstacles(1)
controller = Controller(car)

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
                obstacles = generate_obstacles(2)
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