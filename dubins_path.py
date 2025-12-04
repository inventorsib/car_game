import numpy as np
from math import cos, sin, atan2, pi, acos, sqrt
from typing import Tuple, List, Optional

class Params:
    """Parameters for Dubins path"""
    def __init__(self, d: List[int]):
        self.d = d  # direction
        self.t1 = [0, 0, 0]  # tangent point 1
        self.t2 = [0, 0, 0]  # tangent point 2
        self.c1 = [0, 0]  # center 1
        self.c2 = [0, 0]  # center 2
        self.len = 0  # total length

class DubinsPath:
    """
    Calculate Dubins paths between two configurations
    Supports LSL, LSR, RSL, RSR, LRL, RLR paths
    """
    
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
            'RSR': [-1, -1],
            'LRL': [1, -1, 1],
            'RLR': [-1, 1, -1]
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
        for method in [self._LSL, self._LSR, self._RSL, self._RSR, self._LRL, self._RLR]:
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

    def _LRL(self) -> Optional[Params]:
        """Left-Right-Left path (three arcs)"""
        lrl = Params(self.direction['LRL'])
        
        cline = self.lc2 - self.lc1
        distance = np.linalg.norm(cline)
        
        if distance > 4 * self.r or distance < 1e-10:
            return None
            
        # Calculate middle circle center
        alpha = acos(distance / (4 * self.r))
        beta = atan2(cline[1], cline[0])
        
        # Middle circle center (right turn between two left turns)
        middle_center = self.lc1 + 2 * self.r * np.array([cos(beta + alpha), sin(beta + alpha)])
        
        # Calculate tangent points
        # First tangent point between first left arc and middle right arc
        v1 = middle_center - self.lc1
        theta1 = atan2(v1[1], v1[0]) - alpha
        
        t1 = self.transform(self.lc1[0], self.lc1[1], self.r, 0, theta1, 1)
        
        # Second tangent point between middle right arc and final left arc
        v2 = self.lc2 - middle_center
        theta2 = atan2(v2[1], v2[0]) - alpha
        
        t2 = self.transform(self.lc2[0], self.lc2[1], self.r, 0, theta2, 1)
        
        # Calculate lengths of three arcs
        # First arc (left)
        v_start = self.s - self.lc1
        v_t1 = t1 - self.lc1
        delta_theta1 = self.directional_theta(v_start, v_t1, 1)
        arc1_len = abs(delta_theta1 * self.r)
        
        # Second arc (right)
        v_t1_middle = t1 - middle_center
        v_t2_middle = t2 - middle_center
        delta_theta_middle = self.directional_theta(v_t1_middle, v_t2_middle, -1)
        arc_middle_len = abs(delta_theta_middle * self.r)
        
        # Third arc (left)
        v_t2_end = t2 - self.lc2
        v_end = self.e - self.lc2
        delta_theta2 = self.directional_theta(v_t2_end, v_end, 1)
        arc2_len = abs(delta_theta2 * self.r)
        
        lrl.t1 = t1.tolist() + [theta1]
        lrl.t2 = t2.tolist() + [theta2]
        lrl.c1 = self.lc1.tolist()
        lrl.c2 = self.lc2.tolist()
        lrl.middle_center = middle_center.tolist()  # Additional center for middle arc
        lrl.len = arc1_len + arc_middle_len + arc2_len
        
        return lrl

    def _RLR(self) -> Optional[Params]:
        """Right-Left-Right path (three arcs)"""
        rlr = Params(self.direction['RLR'])
        
        cline = self.rc2 - self.rc1
        distance = np.linalg.norm(cline)
        
        if distance > 4 * self.r or distance < 1e-10:
            return None
            
        # Calculate middle circle center
        alpha = acos(distance / (4 * self.r))
        beta = atan2(cline[1], cline[0])
        
        # Middle circle center (left turn between two right turns)
        middle_center = self.rc1 + 2 * self.r * np.array([cos(beta - alpha), sin(beta - alpha)])
        
        # Calculate tangent points
        # First tangent point between first right arc and middle left arc
        v1 = middle_center - self.rc1
        theta1 = atan2(v1[1], v1[0]) + alpha
        
        t1 = self.transform(self.rc1[0], self.rc1[1], self.r, 0, theta1, 2)
        
        # Second tangent point between middle left arc and final right arc
        v2 = self.rc2 - middle_center
        theta2 = atan2(v2[1], v2[0]) + alpha
        
        t2 = self.transform(self.rc2[0], self.rc2[1], self.r, 0, theta2, 2)
        
        # Calculate lengths of three arcs
        # First arc (right)
        v_start = self.s - self.rc1
        v_t1 = t1 - self.rc1
        delta_theta1 = self.directional_theta(v_start, v_t1, -1)
        arc1_len = abs(delta_theta1 * self.r)
        
        # Second arc (left)
        v_t1_middle = t1 - middle_center
        v_t2_middle = t2 - middle_center
        delta_theta_middle = self.directional_theta(v_t1_middle, v_t2_middle, 1)
        arc_middle_len = abs(delta_theta_middle * self.r)
        
        # Third arc (right)
        v_t2_end = t2 - self.rc2
        v_end = self.e - self.rc2
        delta_theta2 = self.directional_theta(v_t2_end, v_end, -1)
        arc2_len = abs(delta_theta2 * self.r)
        
        rlr.t1 = t1.tolist() + [theta1]
        rlr.t2 = t2.tolist() + [theta2]
        rlr.c1 = self.rc1.tolist()
        rlr.c2 = self.rc2.tolist()
        rlr.middle_center = middle_center.tolist()  # Additional center for middle arc
        rlr.len = arc1_len + arc_middle_len + arc2_len
        
        return rlr

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
        
        # For three-segment paths (LRL, RLR)
        if len(params.d) == 3:
            # First arc
            arc1_points = self._generate_arc_points(
                self.start_pos, params.t1, params.d[0], params.c1, num_points // 3
            )
            points.extend(arc1_points)
            
            # Middle arc
            if hasattr(params, 'middle_center'):
                arc_middle_points = self._generate_arc_points(
                    params.t1, params.t2, params.d[1], params.middle_center, num_points // 3
                )
                points.extend(arc_middle_points[1:])  # Avoid duplicate point
            
            # Final arc
            arc2_points = self._generate_arc_points(
                params.t2, self.end_pos, params.d[2], params.c2, num_points // 3
            )
            points.extend(arc2_points[1:])  # Avoid duplicate point
            
        else:
            # Two-segment paths with straight line (LSL, LSR, RSL, RSR)
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
        
        # Убедимся, что num_points минимум 1
        if num_points < 1:
            num_points = 1
        
        start_angle = atan2(start[1] - center[1], start[0] - center[0])
        end_angle = atan2(end[1] - center[1], end[0] - center[0])
        
        # Normalize angles to [0, 2π)
        start_angle = start_angle #// % (2 * pi)
        end_angle = end_angle #// % (2 * pi)
        
        # Calculate angular difference based on direction
        # if direction == 1:  # Left turn (counterclockwise)
        #     if end_angle < start_angle:
        #         end_angle += 2 * pi
        #     angle_diff = end_angle - start_angle
        # else:  # Right turn (clockwise)
        #     if end_angle > start_angle:
        #         start_angle += 2 * pi
        #     angle_diff = start_angle - end_angle
        #     # For right turn, we need to go backwards
        #     start_angle, end_angle = end_angle, start_angle
        #     angle_diff = -angle_diff
        
        angle_diff = end_angle - start_angle

        # Ensure we don't have zero points
        if num_points == 0:
            num_points = 1
        
        for i in range(num_points + 1):
            t = i / num_points
            angle = start_angle + angle_diff * t
            x = center[0] + self.r * cos(angle)
            y = center[1] + self.r * sin(angle)
            
            # Calculate orientation (tangent to circle)
            if direction == 1:  # Left turn
                theta = angle + pi / 2
            else:  # Right turn
                theta = angle - pi / 2
                
            # Normalize theta to [0, 2π)
            theta = theta #//% (2 * pi)
            points.append((x, y, theta))
            
        return points

    def _generate_straight_points(self, start: List[float] = None, 
                                end: List[float] = None, 
                                num_points: int = 50) -> List[Tuple[float, float, float]]:
        """Generate points along a straight line"""
        if start is None:
            start = self.start_pos
        if end is None:
            end = self.end_pos
        
        # Ensure num_points is at least 1
        if num_points < 1:
            num_points = 1
        
        points = []
        # Calculate orientation difference considering angle wrapping
        theta_start = start[2]
        theta_end = end[2]
        
        # Handle angle wrapping for interpolation
        angle_diff = theta_end - theta_start
        if angle_diff > pi:
            angle_diff -= 2 * pi
        elif angle_diff < -pi:
            angle_diff += 2 * pi
        
        for i in range(num_points + 1):
            t = i / num_points
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            theta = theta_start + angle_diff * t
            # Normalize to [0, 2π)
            theta = theta #//% (2 * pi)
            points.append((x, y, theta))
        return points