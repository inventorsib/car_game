import numpy as np
import math
import heapq
from typing import Tuple, List, Optional, Dict
import matplotlib.pyplot as plt
from math import tan, atan2, acos, pi, sqrt, cos, sin


class Node:
    """Узел для Hybrid A*"""
    def __init__(self, x: float, y: float, theta: float, 
                 g: float = 0, h: float = 0, previousNode=None,
                 analytic_path: List[Tuple[float, float, float]] = None):
        self.x = x
        self.y = y
        self.theta = theta  # ориентация в радианах
        self.g = g  # стоимость от старта
        self.h = h  # эвристическая стоимость до цели
        self.previousNode = previousNode
        # Храним промежуточные точки аналитического пути
        self.analytic_path = analytic_path if analytic_path else []
        
    @property
    def f(self):
        return self.g + self.h
        
    def __lt__(self, other):
        return self.f < other.f

class CachedDubinsHeuristic:
    def __init__(self, turning_radius: float, cache_size: int = 10000):
        self.dubins = DubinsPath(turning_radius)
        self.cache: Dict[Tuple, float] = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0
        
    def _create_cache_key(self, start: Tuple[float, float, float], 
                         goal: Tuple[float, float, float]) -> Tuple:
        """Создание ключа для кэша с дискретизацией"""
        # Дискретизация координат и углов для группировки похожих запросов
        x1 = round(start[0] * 2) / 20  # дискретизация до 0.5 метра
        y1 = round(start[1] * 2) / 20
        theta1 = round(start[2] / (np.pi / 40))  # дискретизация до pi/20 радиан
        
        x2 = round(goal[0] * 2) / 20
        y2 = round(goal[1] * 2) / 20 
        theta2 = round(goal[2] / (np.pi / 40)) 
        
        return (x1, y1, theta1, x2, y2, theta2)
    
    def get_heuristic(self, start: Tuple[float, float, float], 
                     goal: Tuple[float, float, float]) -> float:
        """Получение эвристики с кэшированием"""
        cache_key = self._create_cache_key(start, goal)
        
        if cache_key in self.cache:
            self.hits += 1
            return self.cache[cache_key]
        
        # Вычисление и кэширование
        self.misses += 1
        length, _, _ = self.dubins.find_shortest_path(start, goal)
        
        # Ограничение размера кэша (LRU-подобное поведение)
        if len(self.cache) >= self.cache_size:
            # Удаляем случайный элемент (упрощенная версия)
            key_to_remove = next(iter(self.cache))
            del self.cache[key_to_remove]
        
        self.cache[cache_key] = length
        return length
    
    def get_stats(self) -> Dict:
        """Статистика использования кэша"""
        hit_ratio = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_ratio': hit_ratio,
            'cache_size': len(self.cache)
        }
    
    def clear_cache(self):
        """Очистка кэша"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

class CollisionChecker:
    def __init__(self, grid: np.ndarray, resolution: float, cache_size: int = 50000):
        self.grid = grid
        self.resolution = resolution
        self.height, self.width = grid.shape
        
        # Кэш для проверки столкновений
        self.collision_cache: Dict[Tuple[int, int], bool] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
    def is_collision(self, x: float, y: float) -> bool:
        """Проверка столкновений с кэшированием"""
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)
        
        cache_key = (grid_x, grid_y)
        
        if cache_key in self.collision_cache:
            self.cache_hits += 1
            return self.collision_cache[cache_key]
        
        self.cache_misses += 1
        is_collision = False
        
        if (0 <= grid_x < self.width and 0 <= grid_y < self.height):
            is_collision = self.grid[grid_y, grid_x] == 1
        else:
            is_collision = True  # Вне карты - столкновение
        
        # Кэшируем результат
        self.collision_cache[cache_key] = is_collision
        
        return is_collision
    
    def batch_check_collisions(self, points: List[Tuple[float, float]]) -> List[bool]:
        """Пакетная проверка столкновений для нескольких точек"""
        results = []
        uncached_points = []
        uncached_indices = []
        
        # Проверяем кэш для всех точек
        for i, (x, y) in enumerate(points):
            grid_x = int(x / self.resolution)
            grid_y = int(y / self.resolution)
            cache_key = (grid_x, grid_y)
            
            if cache_key in self.collision_cache:
                results.append(self.collision_cache[cache_key])
            else:
                results.append(None)  # Заполнитель
                uncached_points.append((x, y))
                uncached_indices.append(i)
        
        # Проверяем некэшированные точки
        for idx, (x, y) in zip(uncached_indices, uncached_points):
            grid_x = int(x / self.resolution)
            grid_y = int(y / self.resolution)
            cache_key = (grid_x, grid_y)
            
            is_collision = False
            if (0 <= grid_x < self.width and 0 <= grid_y < self.height):
                is_collision = self.grid[grid_y, grid_x] == 1
            else:
                is_collision = True
                
            self.collision_cache[cache_key] = is_collision
            results[idx] = is_collision
        
        return results

class SuccessorCache:
    def __init__(self, cache_size: int = 10000):
        self.cache: Dict[Tuple, List[Node]] = {}
        self.cache_size = cache_size
        self.hits = 0
        self.misses = 0
        
    def _create_node_key(self, node: Node, angle_resolution: float) -> Tuple:
        """Создание ключа для узла"""
        x = round(node.x * 2) / 2  # дискретизация
        y = round(node.y * 2) / 2
        theta = round(node.theta / angle_resolution)
        return (x, y, theta)
    
    def get_successors(self, node: Node, angle_resolution: float, 
                      generator_func) -> List[Node]:
        """Получение преемников с кэшированием"""
        cache_key = self._create_node_key(node, angle_resolution)
        
        if cache_key in self.cache:
            self.hits += 1
            # Возвращаем копии узлов (чтобы не менять кэшированные)
            cached_successors = self.cache[cache_key]
            return self._copy_successors(cached_successors, node)
        
        self.misses += 1
        # Генерируем преемников
        successors = generator_func(node)
        
        # Кэшируем (ограничиваем размер)
        if len(self.cache) < self.cache_size:
            self.cache[cache_key] = successors
        
        return successors
    
    def _copy_successors(self, successors: List[Node], parent: Node) -> List[Node]:
        """Создание копий преемников с обновлением родителя"""
        copied = []
        for succ in successors:
            # Создаем новый узел с теми же координатами, но обновляем родителя
            new_node = Node(succ.x, succ.y, succ.theta, succ.g, succ.h, parent)
            copied.append(new_node)
        return copied

class Params:
    """Store parameters for different dubins paths."""

    def __init__(self, d):
        self.d = d      # dubins type
        self.t1 = None  # first tangent point
        self.t2 = None  # second tangent point
        self.c1 = None  # first center point
        self.c2 = None  # second center point
        self.len = None # total travel distance

class DubinsPath:
    """
    Calculate Dubins paths between two configurations
    Supports LSL, LSR, RSL, RSR paths
    """
    
    # TODO: LRL, RLR configurations - check
    
    def __init__(self, turning_radius: float = 5.0):
        """
        Args:
            turning_radius: minimum turning radius of the vehicle
        """
        self.r = turning_radius
        
        # turn left: 1, turn right: -1
        self.direction = {
            'LSL': [1, 1],
            'LSR': [1, -1],
            'RSL': [-1, 1],
            'RSR': [-1, -1]
        }

    def transform(self, x: float, y: float, dx: float, dy: float, theta: float, mode: int) -> np.ndarray:
        """
        Transform point based on mode
        mode 1: left transform, mode 2: right transform
        """
        if mode == 1:  # left
            new_x = x + dx * cos(theta) - dy * sin(theta)
            new_y = y + dx * sin(theta) + dy * cos(theta)
        else:  # right
            new_x = x + dx * cos(theta) + dy * sin(theta)
            new_y = y + dx * sin(theta) - dy * cos(theta)
        return np.array([new_x, new_y])

    def directional_theta(self, v1: np.ndarray, v2: np.ndarray, direction: int) -> float:
        """Calculate directional angle between two vectors"""
        dot_product = np.dot(v1, v2)
        det = v1[0] * v2[1] - v1[1] * v2[0]
        angle = atan2(det, dot_product)
        
        # Adjust angle based on direction
        if direction == -1 and angle > 0:
            angle -= 2 * pi
        elif direction == 1 and angle < 0:
            angle += 2 * pi
            
        return angle

    def find_shortest_path(self, start_pos: Tuple[float, float, float], 
                          end_pos: Tuple[float, float, float]) -> Tuple[float, str, Optional[Params]]:
        """
        Find the shortest Dubins path between start and end configurations
        
        Args:
            start_pos: (x, y, theta) start configuration
            end_pos: (x, y, theta) end configuration
            
        Returns:
            (length, path_type, params) - path length, type and parameters
        """
        self.start_pos = start_pos
        self.end_pos = end_pos

        x1, y1, theta1 = start_pos
        x2, y2, theta2 = end_pos
        
        self.s = np.array(start_pos[:2])
        self.e = np.array(end_pos[:2])
        
        # Calculate center points for left and right turns
        self.lc1 = self.transform(x1, y1, 0, self.r, theta1, 1)  # left center start
        self.rc1 = self.transform(x1, y1, 0, self.r, theta1, 2)  # right center start
        self.lc2 = self.transform(x2, y2, 0, self.r, theta2, 1)  # left center end
        self.rc2 = self.transform(x2, y2, 0, self.r, theta2, 2)  # right center end
        
        # Calculate all possible paths
        solutions = []
        for method in [self._LSL, self._LSR, self._RSL, self._RSR]:
            try:
                solution = method()
                if solution is not None and solution.len >= 0:
                    solutions.append(solution)
            except (ValueError, ZeroDivisionError):
                continue
        
        if not solutions:
            # Fallback to straight line distance
            straight_dist = np.linalg.norm(self.e - self.s)
            return straight_dist, "STRAIGHT", None
        
        # Find shortest path
        best_solution = min(solutions, key=lambda x: x.len)
        path_type = self._get_path_type(best_solution.d)
        
        return best_solution.len, path_type, best_solution

    def _get_path_type(self, direction: List[int]) -> str:
        """Convert direction list to path type string"""
        for key, value in self.direction.items():
            if value == direction:
                return key
        return "UNKNOWN"

    def _LSL(self) -> Optional[Params]:
        """Left-Straight-Left path"""
        lsl = Params(self.direction['LSL'])
        
        cline = self.lc2 - self.lc1
        distance = np.linalg.norm(cline)
        
        # Критическое исправление: правильный расчет угла
        if distance < 2 * self.r:  # Случай перекрывающихся окружностей
            return None
            
        theta = atan2(cline[1], cline[0])
        alpha = acos(2 * self.r / distance) if distance > 2 * self.r else 0

        t1 = self.transform(self.lc1[0], self.lc1[1], self.r, 0, theta + alpha, 1)
        t2 = self.transform(self.lc2[0], self.lc2[1], self.r, 0, theta + alpha, 1)
        
        return self._get_params(lsl, self.lc1, self.lc2, t1, t2)

    def _LSR(self) -> Optional[Params]:
        """Left-Straight-Right path"""
        lsr = Params(self.direction['LSR'])

        cline = self.rc2 - self.lc1
        R = np.linalg.norm(cline) / 2

        if R < self.r or R < 1e-10:
            return None
        
        theta = atan2(cline[1], cline[0]) - acos(self.r / R)

        t1 = self.transform(self.lc1[0], self.lc1[1], self.r, 0, theta, 1)
        t2 = self.transform(self.rc2[0], self.rc2[1], self.r, 0, theta + pi, 1)

        return self._get_params(lsr, self.lc1, self.rc2, t1, t2)

    def _RSL(self) -> Optional[Params]:
        """Right-Straight-Left path"""
        rsl = Params(self.direction['RSL'])

        cline = self.lc2 - self.rc1
        R = np.linalg.norm(cline) / 2

        if R < self.r or R < 1e-10:
            return None
        
        theta = atan2(cline[1], cline[0]) + acos(self.r / R)

        t1 = self.transform(self.rc1[0], self.rc1[1], self.r, 0, theta, 1)
        t2 = self.transform(self.lc2[0], self.lc2[1], self.r, 0, theta + pi, 1)

        return self._get_params(rsl, self.rc1, self.lc2, t1, t2)

    def _RSR(self) -> Optional[Params]:
        """Right-Straight-Right path"""
        rsr = Params(self.direction['RSR'])

        cline = self.rc2 - self.rc1
        R = np.linalg.norm(cline) / 2
        
        if R < 1e-10:
            return None
            
        theta = atan2(cline[1], cline[0]) + acos(0)

        t1 = self.transform(self.rc1[0], self.rc1[1], self.r, 0, theta, 1)
        t2 = self.transform(self.rc2[0], self.rc2[1], self.r, 0, theta, 1)

        return self._get_params(rsr, self.rc1, self.rc2, t1, t2)

    def _get_params(self, dub: Params, c1: np.ndarray, c2: np.ndarray, 
                   t1: np.ndarray, t2: np.ndarray) -> Params:
        """Calculate the dubins path parameters and length"""
        
        v1 = self.s - c1
        v2 = t1 - c1
        v3 = t2 - t1
        v4 = t2 - c2
        v5 = self.e - c2

        delta_theta1 = self.directional_theta(v1, v2, dub.d[0])
        delta_theta2 = self.directional_theta(v4, v5, dub.d[1])

        arc1 = abs(delta_theta1 * self.r)
        tangent = np.linalg.norm(v3)
        arc2 = abs(delta_theta2 * self.r)

        theta = self.start_pos[2] + delta_theta1

        dub.t1 = t1.tolist() + [theta]
        dub.t2 = t2.tolist() + [theta]
        dub.c1 = c1.tolist()
        dub.c2 = c2.tolist()
        dub.len = arc1 + tangent + arc2
        
        return dub

    def generate_path_points(self, params: Params, num_points: int = 100) -> List[Tuple[float, float, float]]:
        """
        Generate points along the Dubins path for visualization
        """
        if params is None:
            # Straight line fallback
            return self._generate_straight_points()
            
        points = []
        
        # First arc
        arc1_points = self._generate_arc_points(
            self.start_pos, params.t1, params.d[0], params.c1, num_points // 3
        )
        points.extend(arc1_points)
        
        # Straight segment
        straight_points = self._generate_straight_points(
            params.t1, params.t2, num_points // 3
        )
        points.extend(straight_points[1:])  # Avoid duplicate point
        
        # Second arc
        arc2_points = self._generate_arc_points(
            params.t2, self.end_pos, params.d[1], params.c2, num_points // 3
        )
        points.extend(arc2_points[1:])  # Avoid duplicate point
        
        return points

    def _generate_arc_points(self, start: List[float], end: List[float], 
                           direction: int, center: List[float], num_points: int) -> List[Tuple[float, float, float]]:
        """Generate points along an arc"""
        points = []
        
        start_angle = atan2(start[1] - center[1], start[0] - center[0])
        end_angle = atan2(end[1] - center[1], end[0] - center[0])
        
        # Adjust angles for direction
        if direction == 1 and end_angle < start_angle:
            end_angle += 2 * pi
        elif direction == -1 and end_angle > start_angle:
            end_angle -= 2 * pi
            
        for i in range(num_points + 1):
            t = i / num_points
            angle = start_angle + (end_angle - start_angle) * t
            x = center[0] + self.r * cos(angle)
            y = center[1] + self.r * sin(angle)
            
            # Calculate orientation (tangent to circle)
            if direction == 1:  # Left turn
                theta = angle + pi / 2
            else:  # Right turn
                theta = angle - pi / 2
                
            points.append((x, y, theta % (2 * pi)))
            
        return points

    def _generate_straight_points(self, start: List[float] = None, 
                                end: List[float] = None, 
                                num_points: int = 50) -> List[Tuple[float, float, float]]:
        """Generate points along a straight line"""
        if start is None:
            start = self.start_pos
        if end is None:
            end = self.end_pos
            
        points = []
        for i in range(num_points + 1):
            t = i / num_points
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            theta = start[2] + (end[2] - start[2]) * t
            points.append((x, y, theta % (2 * pi)))
        return points


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
        self.turning_radius = self.wheel_base / math.tan(self.max_steering_angle)
        
        # Параметры аналитического расширения
        self.expansion_interval = 1  # Каждый N-й узел пытаемся сделать расширение
        self.max_analytic_expansion_distance = 200.0  # Макс. расстояние для расширения
        self.analytic_expansion_max_points = 50  # Макс точек в аналитическом пути
        self.min_analytic_segment_length = 0.1   # Минимальная длина сегмента

        # Эвристика Дубинса
        self.dubins = DubinsPath(self.turning_radius)

        # Дискретизация углов
        self.angle_resolution = math.radians(1)

        # Системы кэширования
        self.dubins_heuristic = CachedDubinsHeuristic(self.turning_radius)
        self.collision_checker = CollisionChecker(grid, resolution)
        self.successor_cache = SuccessorCache()

    
    def heuristic(self, node: Node, goal: Node):
        """Эвристика с кэшированием"""
        start_config = (node.x, node.y, node.theta)
        goal_config = (goal.x, goal.y, goal.theta)
        return self.dubins_heuristic.get_heuristic(start_config, goal_config)
    
    def is_collision(self, x: float, y: float):
        """Проверка столкновений с кэшированием"""
        return self.collision_checker.is_collision(x, y)
    
    def get_successors(self, node: Node):
        """Генерация преемников с кэшированием"""
        return self.successor_cache.get_successors(
            node, self.angle_resolution, self._generate_successors_impl
        )
    
    def _generate_successors_impl(self, node: Node) -> List[Node]:
        """Реальная генерация преемников (без кэширования)"""
        successors = []
        steering_angles = [-self.max_steering_angle, 0, self.max_steering_angle]
        
        # Адаптивные шаги
        if node.h < 5.0:
            step_sizes = [0.1, 0.25, 0.5]
        else:
            step_sizes = [1.0, 2.0, 3.0]

        step_sizes = [0.25, 0.5, 1.0]

        # Пакетная проверка столкновений для всех потенциальных преемников
        potential_points = []
        point_to_params = {}  # маппинг точек к параметрам
        
        for steering in steering_angles:
            for step in step_sizes:
                if abs(steering) < 1e-5:
                    new_x = node.x + step * math.cos(node.theta)
                    new_y = node.y + step * math.sin(node.theta)
                    new_theta = node.theta
                else:
                    turning_radius = self.wheel_base / math.tan(steering)
                    angular_velocity = step / turning_radius
                    new_theta = node.theta + angular_velocity
                    new_x = node.x + turning_radius * (math.sin(new_theta) - math.sin(node.theta))
                    new_y = node.y + turning_radius * (math.cos(node.theta) - math.cos(new_theta))
                
                new_theta = new_theta % (2 * math.pi)
                potential_points.append((new_x, new_y))
                point_to_params[(new_x, new_y)] = (steering, step, new_theta)
        
        # Пакетная проверка столкновений
        collision_results = self.collision_checker.batch_check_collisions(potential_points)
        
        # Создаем преемников только для свободных точек
        for (x, y), is_collision in zip(potential_points, collision_results):
            if not is_collision:
                steering, step, theta = point_to_params[(x, y)]
                cost = step
                successor = Node(x, y, theta, node.g + cost, 0, node)
                successors.append(successor)
        
        return successors
    
    def print_cache_stats(self):
        """Вывод статистики кэширования"""
        dubins_stats = self.dubins_heuristic.get_stats()
        collision_stats = {
            'hits': self.collision_checker.cache_hits,
            'misses': self.collision_checker.cache_misses,
            'hit_ratio': self.collision_checker.cache_hits / 
                        (self.collision_checker.cache_hits + self.collision_checker.cache_misses) 
                        if (self.collision_checker.cache_hits + self.collision_checker.cache_misses) > 0 else 0
        }
        successor_stats = {
            'hits': self.successor_cache.hits,
            'misses': self.successor_cache.misses,
            'hit_ratio': self.successor_cache.hits / 
                        (self.successor_cache.hits + self.successor_cache.misses) 
                        if (self.successor_cache.hits + self.successor_cache.misses) > 0 else 0
        }
        
        print("=== КЭШ СТАТИСТИКА ===")
        print(f"Дубенс эвристика: {dubins_stats['hit_ratio']:.1%} ({dubins_stats['hits']}/{dubins_stats['misses']})")
        print(f"Проверка столкновений: {collision_stats['hit_ratio']:.1%} ({collision_stats['hits']}/{collision_stats['misses']})")
        print(f"Генерация преемников: {successor_stats['hit_ratio']:.1%} ({successor_stats['hits']}/{successor_stats['misses']})")



    def discretize_state(self, node: Node) -> Tuple[int, int, int]:
        """Дискретизация состояния для проверки посещённых узлов"""
        x_idx = int(node.x / self.resolution)
        y_idx = int(node.y / self.resolution)
        theta_idx = int(node.theta / self.angle_resolution)
        return (x_idx, y_idx, theta_idx)
    
    def search(self, start: Tuple[float, float, float], 
               goal: Tuple[float, float, float]) -> List[Node]:
        start_node = Node(start[0], start[1], start[2])
        goal_node = Node(goal[0], goal[1], goal[2])
        
        start_node.h = self.heuristic(start_node, goal_node)
        
        open_list = []
        heapq.heappush(open_list, (start_node.f, start_node))
        
        closed_set = set()
        visited = {}
        
        node_counter = 0  # Счетчик для отслеживания интервала расширения
        
        while open_list:
            _, current = heapq.heappop(open_list)
            
            # Проверка достижения цели
            if self.is_goal_reached(current, goal_node):
                return self.reconstruct_path(current)
            
            current_state = self.discretize_state(current)
            if current_state in closed_set:
                continue
                
            closed_set.add(current_state)
            node_counter += 1
            
            # АНАЛИТИЧЕСКОЕ РАСШИРЕНИЕ - пробуем каждый N-й узел
            if node_counter % self.expansion_interval == 0:
                analytic_node = self.analytic_expansion(current, goal_node)
                if analytic_node and not self.is_collision(analytic_node.x, analytic_node.y):
                    # Нашли прямой путь до цели!
                    return self.reconstruct_path(analytic_node)
            
            # Обычная генерация преемников
            for successor in self.get_successors(current):
                successor_state = self.discretize_state(successor)
                
                if successor_state in closed_set:
                    continue
                
                successor.h = self.heuristic(successor, goal_node)
                
                if successor_state in visited and visited[successor_state] <= successor.g:
                    continue
                
                visited[successor_state] = successor.g
                heapq.heappush(open_list, (successor.f, successor))
        
        return []
    
    def is_goal_reached(self, node: Node, goal: Node, pos_tolerance: float = 0.5, 
                       angle_tolerance: float = math.radians(15)) -> bool:
        """Проверка достижения цели"""
        dx = goal.x - node.x
        dy = goal.y - node.y
        distance = math.sqrt(dx**2 + dy**2)
        
        angle_diff = min(abs(node.theta - goal.theta), 
                        2 * math.pi - abs(node.theta - goal.theta))
        
        return distance <= pos_tolerance and angle_diff <= angle_tolerance
    

    def analytic_expansion(self, node: Node, goal: Node) -> Optional[Node]:
        """
        Улучшенное аналитическое расширение с генерацией промежуточных точек
        """
        distance_to_goal = math.sqrt((goal.x - node.x)**2 + (goal.y - node.y)**2)
        
        if distance_to_goal > self.max_analytic_expansion_distance:
            return None
        
        start_config = (node.x, node.y, node.theta)
        goal_config = (goal.x, goal.y, goal.theta)
        
        # Получаем путь Дубенса
        length, path_type, params = self.dubins.find_shortest_path(
            start_config, goal_config
        )
        
        if params is None or length > self.max_analytic_expansion_distance * 1.5:
            return None
        
        # Проверяем, что путь свободен от препятствий
        if not self.is_dubins_path_collision_free(params):
            return None
        
        # Генерируем промежуточные точки пути Дубенса
        analytic_points = self.generate_analytic_path_points(node, goal, params)
        
        if not analytic_points:
            return None
            
        # Создаем конечный узел с прикрепленным аналитическим путем
        analytic_node = Node(goal.x, goal.y, goal.theta, 
                           node.g + length, 0, node, analytic_points)
        
        return analytic_node
    
    def generate_analytic_path_points(self, start_node: Node, goal_node: Node, 
                                    params: Params) -> List[Tuple[float, float, float]]:
        """
        Генерирует промежуточные точки аналитического пути Дубенса
        с проверкой столкновений для каждого сегмента
        """
        # Генерируем точки вдоль пути Дубенса
        dubins_points = self.dubins.generate_path_points(
            params, self.analytic_expansion_max_points
        )
        
        # Фильтруем точки: оставляем только те, что проходят проверку столкновений
        safe_points = []
        previous_point = (start_node.x, start_node.y, start_node.theta)
        
        for point in dubins_points:
            x, y, theta = point
            
            # Проверяем столкновение для текущей точки
            if self.is_collision(x, y):
                break  # Прерываем если наткнулись на препятствие
                
            # Проверяем весь сегмент от предыдущей точки до текущей
            if not self.is_segment_collision_free(previous_point, point):
                break
                
            safe_points.append(point)
            previous_point = point
        
        # Всегда добавляем конечную точку (цель)
        if not safe_points or safe_points[-1] != (goal_node.x, goal_node.y, goal_node.theta):
            if not self.is_collision(goal_node.x, goal_node.y):
                safe_points.append((goal_node.x, goal_node.y, goal_node.theta))
        
        return safe_points
    
    def is_segment_collision_free(self, start: Tuple[float, float, float], 
                                end: Tuple[float, float, float], 
                                step_size: float = 0.5) -> bool:
        """
        Проверяет весь сегмент пути на столкновения
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < self.min_analytic_segment_length:
            return True
            
        steps = max(2, int(distance / step_size))
        
        for i in range(1, steps):
            t = i / steps
            x = start[0] + dx * t
            y = start[1] + dy * t
            
            if self.is_collision(x, y):
                return False
                
        return True

    def is_dubins_path_collision_free(self, params: Params, step_size: float = 0.5) -> bool:
        """
        Проверка столкновений вдоль пути Дубенса
        """
        # Генерируем точки вдоль пути с мелким шагом
        points = self.dubins.generate_path_points(params, 
                                                           int(params.len / step_size))
        
        for point in points:
            x, y, theta = point
            if self.is_collision(x, y):
                return False
        return True
    
    def create_node_from_dubins(self, start_node: Node, params: Params) -> Node:
        """
        Создание узлов из пути Дубенса для более гладкого пути
        """
        points = self.dubins_heuristic.generate_path_points(params, 10)
        
        current = start_node
        for point in points[1:]:  # Пропускаем стартовую точку
            x, y, theta = point
            # Вычисляем длину сегмента
            segment_length = math.sqrt((x - current.x)**2 + (y - current.y)**2)
            
            new_node = Node(x, y, theta, 
                          current.g + segment_length, 0, current)
            current = new_node
        
        return current


    def reconstruct_path(self, node: Node) -> List[Node]:
        """Восстановление пути от конечного узла до старта"""
        path = []
        current = node
        while current:
            path.append(current)
            current = current.previousNode
        return path[::-1]
    

def visualize_path_with_analytic(grid: np.ndarray, path: List[Node], start, goal):
    """Визуализация пути с выделением аналитических сегментов"""
    plt.figure(figsize=(14, 12))
    
    # Отображение карты
    plt.imshow(grid, cmap='Greys', origin='lower')
    
    if path:
        # Разделяем обычные точки и аналитические сегменты
        regular_x, regular_y = [], []
        analytic_segments = []
        
        current_segment = []
        for node in path:
            if node.analytic_path:
                # Начало аналитического сегмента
                if current_segment:
                    analytic_segments.append(current_segment)
                    current_segment = []
                
                # Добавляем аналитический путь
                segment_x = [p[0] for p in node.analytic_path]
                segment_y = [p[1] for p in node.analytic_path]
                analytic_segments.append(list(zip(segment_x, segment_y)))
            else:
                # Обычная точка
                regular_x.append(node.x)
                regular_y.append(node.y)
                current_segment.append((node.x, node.y))
        
        if current_segment:
            analytic_segments.append(current_segment)
        
        # Рисуем обычные точки
        plt.plot(regular_x, regular_y, 'bo', markersize=3, alpha=0.5, label='Обычные узлы')
        
        # Рисуем аналитические сегменты разными цветами
        colors = ['red', 'green', 'purple', 'orange', 'cyan']
        for i, segment in enumerate(analytic_segments):
            if segment:
                seg_x, seg_y = zip(*segment)
                color = colors[i % len(colors)]
                plt.plot(seg_x, seg_y, color=color, linewidth=3, 
                        label=f'Аналитический сегмент {i+1}')
                plt.plot(seg_x, seg_y, 'o', color=color, markersize=2)
    
    # Старт и цель
    plt.plot(start[0], start[1], 'go', markersize=12, label='Старт', markeredgecolor='black')
    plt.plot(goal[0], goal[1], 'ro', markersize=12, label='Цель', markeredgecolor='black')
    
    # Ориентация
    arrow_length = 2.0
    plt.arrow(start[0], start[1], 
            arrow_length * math.cos(start[2]), arrow_length * math.sin(start[2]),
            head_width=0.8, fc='green', ec='green', linewidth=2)
    plt.arrow(goal[0], goal[1],
            arrow_length * math.cos(goal[2]), arrow_length * math.sin(goal[2]),
            head_width=0.8, fc='red', ec='red', linewidth=2)
    
    plt.legend()
    plt.grid(True)
    plt.title('Hybrid A* Path с аналитическими расширениями')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def print_path_statistics(path: List[Node]):
    """Вывод статистики по пути"""
    if not path:
        print("Путь пуст")
        return
    
    analytic_segments = 0
    analytic_points = 0
    regular_points = 0
    
    for node in path:
        if node.analytic_path:
            analytic_segments += 1
            analytic_points += len(node.analytic_path)
        else:
            regular_points += 1
    
    print("=== СТАТИСТИКА ПУТИ ===")
    print(f"Всего точек: {len(path)}")
    print(f"Обычные узлы: {regular_points}")
    print(f"Аналитические сегменты: {analytic_segments}")
    print(f"Точек в аналитических путях: {analytic_points}")
    print(f"Общая длина пути: {path[-1].g:.2f} м")
    
    if analytic_segments > 0:
        coverage = analytic_points / (regular_points + analytic_points) * 100
        print(f"Покрытие аналитическим путем: {coverage:.1f}%")

    # Пример использования
def create_test_grid():
    """Создание тестовой карты"""
    grid = np.zeros((50, 50))
    
    # Добавление препятствий
    grid[15:25, 0:33] = 1  # прямоугольное препятствие
    #grid[21:35, 15:50] = 1  # прямоугольное препятствие
    #//grid[11:15, 11:30] = 1  # стена
    
    return grid

def visualize_path(grid:np.ndarray, path, start, goal):
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
    
    grid_2d = np.shape(grid)
    plt.xticks(np.arange(0, grid_2d[0], 5))  # шаг 1 от 0 до 10
    plt.yticks(np.arange(0, grid_2d[1], 5))  # шаг 0.2 от -1 до 1

    plt.grid(True) #grid_resolution
    plt.title('Hybrid A* Path Planning')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Демонстрация
if __name__ == "__main__":
    grid = create_test_grid()
    planner = HybridAStar(grid, resolution=1.0)
    
    start = (0.0, 0.0, math.radians(0))
    goal = (40.0, 40.0, math.radians(90))
    
    path = planner.search(start, goal)
    
    if path:
        print(f"Путь найден! Длина: {len(path)} узлов")
        print(f"Общая стоимость: {path[-1].g:.2f}")
        
        # Выводим статистику
        print_path_statistics(path)
        
        # Показываем детали аналитических расширений
        for i, node in enumerate(path):
            if node.analytic_path:
                print(f"Узел {i}: аналитический путь с {len(node.analytic_path)} точками")
        
        # Визуализируем
        visualize_path_with_analytic(grid, path, start, goal)
    else:
        print("Путь не найден!")