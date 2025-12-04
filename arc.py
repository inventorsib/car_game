import matplotlib.pyplot as plt

import math
import numpy as np

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

# Пример использования
if __name__ == "__main__":
    O = (0, 0)
    R = 5
    P = (10, 2)
    R_arc = 4
    
    solutions = generate_tangent_arc_points(O, R, P, R_arc, num_points=20)
    
    for i, (C, T, arc_points) in enumerate(solutions):
        print(f"Решение {i+1}:")
        print(f"  Центр дуги C: ({C[0]:.2f}, {C[1]:.2f})")
        print(f"  Точка касания T: ({T[0]:.2f}, {T[1]:.2f})")
        print(f"  Количество точек дуги: {len(arc_points)}")
        print(f"  Первые 5 точек дуги от P до T:")
        for j, point in enumerate(arc_points[:5]):
            print(f"    Точка {j}: ({point[0]:.2f}, {point[1]:.2f})")
        print()

def plot_tangent_arc(O, R, P, R_arc, num_points=100):
    """Визуализация касательной дуги"""
    
    solutions = generate_tangent_arc_points(O, R, P, R_arc, num_points)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Рисуем исходную окружность
    circle = plt.Circle(O, R, fill=False, color='blue', linestyle='--', linewidth=2)
    ax.add_patch(circle)
    
    # Рисуем начальную точку
    ax.plot(P[0], P[1], 'ro', markersize=8, label=f'Точка P {P}')
    
    colors = ['red', 'green']
    
    for i, (C, T, arc_points) in enumerate(solutions):
        color = colors[i % len(colors)]
        
        # Рисуем дугу
        arc_x = [p[0] for p in arc_points]
        arc_y = [p[1] for p in arc_points]
        ax.plot(arc_x, arc_y, color=color, linewidth=3, label=f'Дуга {i+1}')
        
        # Рисуем центр дуги
        ax.plot(C[0], C[1], 'o', color=color, markersize=6, label=f'Центр C{i+1}')
        
        # Рисуем точку касания
        ax.plot(T[0], T[1], 's', color=color, markersize=8, label=f'Точка касания T{i+1}')
        
        # Рисуем радиус дуги
        circle_arc = plt.Circle(C, R_arc, fill=False, color=color, linestyle=':', alpha=0.5)
        ax.add_patch(circle_arc)
    
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(f'Касательные дуги: R={R}, R_arc={R_arc}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Визуализация
O = (0, 0)
R = 5
P = (10, 2)
R_arc = 4

plot_tangent_arc(O, R, P, R_arc)