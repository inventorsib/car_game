from random import randint
import os
import sys 
import time
from time import sleep

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

    def add_neighbors(self, grid, columns, rows):
        neighbor_x = self.x
        neighbor_y = self.y
    
        if neighbor_x < columns - 1:
            self.neighbors.append(grid[neighbor_x+1][neighbor_y])
        if neighbor_x > 0:
            self.neighbors.append(grid[neighbor_x-1][neighbor_y])
        if neighbor_y < rows -1:
            self.neighbors.append(grid[neighbor_x][neighbor_y +1])
        if neighbor_y > 0: 
            self.neighbors.append(grid[neighbor_x][neighbor_y-1])

class AStar:
    def __init__(self, cols, rows, start, end):
        self.cols = cols
        self.rows = rows
        self.start = start
        self.end = end
        self.obstacle_ratio = False
        self.obstacle_list = False

    @staticmethod
    def clean_open_set(open_set, current_node):
        for i in range(len(open_set)):
            if open_set[i] == current_node:
                open_set.pop(i)
                break
        return open_set

    @staticmethod
    def h_score(current_node, end):
        distance = abs(current_node.x - end.x) + abs(current_node.y - end.y)
        return distance

    @staticmethod
    def create_grid(cols, rows):
        grid = []
        for _ in range(cols):
            grid.append([])
            for _ in range(rows):
                grid[-1].append(0)
        return grid

    @staticmethod
    def fill_grids(grid, cols, rows, obstacle_ratio=False, obstacle_list=False):
        for i in range(cols):
            for j in range(rows):
                grid[i][j] = Node(i,j)
                if obstacle_ratio == False:
                    pass
                else:
                    n = randint(0,100)
                    if n < obstacle_ratio: 
                        grid[i][j].obstacle = True
        if obstacle_list == False:
            pass
        else:
            for i in range(len(obstacle_list)):
                grid[obstacle_list[i][0]][obstacle_list[i][1]].obstacle = True
        return grid

    @staticmethod
    def get_neighbors(grid, cols, rows):
        for i in range(cols):
            for j in range(rows):
                grid[i][j].add_neighbors(grid, cols, rows)
        return grid
    
    @staticmethod
    def start_path(open_set, closed_set, current_node, end):
        best_way = 0
        for i in range(len(open_set)):
            if open_set[i].f < open_set[best_way].f:
                best_way = i

        current_node = open_set[best_way]
        final_path = []
        if current_node == end:
            temp = current_node
            final_path.append(temp)
            while temp.previous:
                final_path.append(temp.previous)
                temp = temp.previous

        open_set = AStar.clean_open_set(open_set, current_node)
        closed_set.append(current_node)
        neighbors = current_node.neighbors
        for neighbor in neighbors:
            if (neighbor in closed_set) or (neighbor.obstacle == True):
                continue
            else:
                temp_g = current_node.g + 1
                control_flag = 0
                for k in range(len(open_set)):
                    if neighbor.x == open_set[k].x and neighbor.y == open_set[k].y:
                        if temp_g < open_set[k].g:
                            open_set[k].g = temp_g
                            open_set[k].h = AStar.h_score(open_set[k], end)
                            open_set[k].f = open_set[k].g + open_set[k].h
                            open_set[k].previous = current_node
                        else:
                            pass
                        control_flag = 1
                if control_flag == 1:
                    pass
                else:
                    neighbor.g = temp_g
                    neighbor.h = AStar.h_score(neighbor, end)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.previous = current_node
                    open_set.append(neighbor)

        return open_set, closed_set, current_node, final_path

    def main(self):
        grid = AStar.create_grid(self.cols, self.rows)
        grid = AStar.fill_grids(grid, self.cols, self.rows, obstacle_ratio=30)
        grid = AStar.get_neighbors(grid, self.cols, self.rows)
        open_set = []
        closed_set = []
        current_node = None
        final_path = []
        open_set.append(grid[self.start[0]][self.start[1]])
        self.end = grid[self.end[0]][self.end[1]]
        
        while len(open_set) > 0:
            open_set, closed_set, current_node, final_path = AStar.start_path(open_set, closed_set, current_node, self.end)
            if len(final_path) > 0:
                break

        return final_path, grid, closed_set

def visualize_path(grid, cols, rows, start, end, final_path, closed_set):
    """
    Визуализация пути в консоли
    """
    print("\n" + "="*50)
    print("ВИЗУАЛИЗАЦИЯ ПУТИ A*")
    print("="*50)
    
    # Создаем символьное представление сетки
    display_grid = []
    for i in range(cols):
        row = []
        for j in range(rows):
            if grid[i][j].obstacle:
                row.append('██')  # Препятствие
            else:
                row.append(' .')  # Свободная клетка
        display_grid.append(row)
    
    # Помечаем старт и финиш
    display_grid[start[0]][start[1]] = ' S'
    display_grid[end[0]][end[1]] = ' E'
    
    # Помечаем посещенные узлы (closed set)
    for node in closed_set:
        if (node.x, node.y) != start and (node.x, node.y) != end and not node.obstacle:
            display_grid[node.x][node.y] = ' *'
    
    # Помечаем финальный путь
    if final_path:
        for node in final_path:
            if (node.x, node.y) != start and (node.x, node.y) != end:
                display_grid[node.x][node.y] = ' ○'
    
    # Выводим сетку
    print("\nЛегенда:")
    print(" S - Старт")
    print(" E - Финиш") 
    print(" ○ - Найденный путь")
    print(" * - Исследованные узлы")
    print(" █ - Препятствие")
    print(" . - Свободное пространство")
    
    print("\nКарта:")
    for j in range(rows):
        for i in range(cols):
            print(display_grid[i][j], end='')
        print()

def print_path_info(final_path, closed_set):
    """
    Вывод информации о найденном пути
    """
    print(f"\nИнформация о пути:")
    print(f"Длина пути: {len(final_path)} шагов")
    print(f"Исследовано узлов: {len(closed_set)}")
    
    if final_path:
        print("\nКоординаты пути:")
        for i, node in enumerate(reversed(final_path)):
            print(f"Шаг {i}: ({node.x}, {node.y})")

def example_1():
    """
    Пример 1: Случайная карта с препятствиями
    """
    print("ПРИМЕР 1: Случайная карта 15x15")
    
    cols, rows = 15, 15
    start = (0, 0)
    end = (14, 14)
    
    astar = AStar(cols, rows, start, end)
    final_path, grid, closed_set = astar.main()
    
    visualize_path(grid, cols, rows, start, end, final_path, closed_set)
    print_path_info(final_path, closed_set)

def example_2():
    """
    Пример 2: Карта с заданными препятствиями
    """
    print("\n" + "="*50)
    print("ПРИМЕР 2: Карта с лабиринтом")
    
    cols, rows = 10, 10
    start = (0, 0)
    end = (9, 9)
    
    # Задаем конкретные препятствия
    obstacle_list = [
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
        (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
        (1, 7), (2, 7), (3, 7), (4, 7), (5, 7),
        (7, 2), (8, 2), (9, 2)
    ]
    
    astar = AStar(cols, rows, start, end)
    
    # Создаем сетку с заданными препятствиями
    grid = AStar.create_grid(cols, rows)
    grid = AStar.fill_grids(grid, cols, rows, obstacle_list=obstacle_list)
    grid = AStar.get_neighbors(grid, cols, rows)
    
    # Запускаем поиск пути
    open_set = []
    closed_set = []
    current_node = None
    final_path = []
    open_set.append(grid[start[0]][start[1]])
    end_node = grid[end[0]][end[1]]
    
    while len(open_set) > 0:
        open_set, closed_set, current_node, final_path = AStar.start_path(open_set, closed_set, current_node, end_node)
        if len(final_path) > 0:
            break
    
    visualize_path(grid, cols, rows, start, end, final_path, closed_set)
    print_path_info(final_path, closed_set)

def example_3():
    """
    Пример 3: Небольшая карта с пошаговой визуализацией
    """
    print("\n" + "="*50)
    print("ПРИМЕР 3: Маленькая карта 8x8")
    
    cols, rows = 8, 8
    start = (1, 1)
    end = (6, 6)
    
    # Меньше препятствий для лучшей визуализации
    astar = AStar(cols, rows, start, end)
    final_path, grid, closed_set = astar.main()
    
    visualize_path(grid, cols, rows, start, end, final_path, closed_set)
    print_path_info(final_path, closed_set)

if __name__ == "__main__":
    # Запускаем примеры
    example_1()
    example_2() 
    example_3()
    
    # Дополнительная информация
    print("\n" + "="*50)
    print("СТАТИСТИКА АЛГОРИТМА")
    print("="*50)
    print("Алгоритм A* находит кратчайший путь от старта до финиша,")
    print("минимизируя стоимость пути (g) + эвристическую оценку (h).")
    print("Эвристика: Манхэттенское расстояние")
    print("Диагональное движение: отключено (можно включить в классе Node)")