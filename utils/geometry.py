"""
Football Analysis - Utilitaires géométriques
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.ndimage import gaussian_filter1d


def calculate_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Calculer la distance euclidienne entre deux points

    Args:
        point1: Premier point (x, y)
        point2: Deuxième point (x, y)

    Returns:
        Distance euclidienne
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def calculate_angle(
    point1: Tuple[float, float],
    point2: Tuple[float, float]
) -> float:
    """
    Calculer l'angle en degrés entre deux points

    Args:
        point1: Point de départ (x, y)
        point2: Point d'arrivée (x, y)

    Returns:
        Angle en degrés (0-360)
    """
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return angle % 360


def calculate_velocity(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    dt: float
) -> Tuple[float, float]:
    """
    Calculer le vecteur vitesse entre deux points

    Args:
        point1: Position initiale (x, y)
        point2: Position finale (x, y)
        dt: Intervalle de temps

    Returns:
        Vecteur vitesse (vx, vy)
    """
    if dt <= 0:
        return (0.0, 0.0)
    return ((point2[0] - point1[0]) / dt, (point2[1] - point1[1]) / dt)


def calculate_speed(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    dt: float
) -> float:
    """
    Calculer la vitesse scalaire entre deux points

    Args:
        point1: Position initiale (x, y)
        point2: Position finale (x, y)
        dt: Intervalle de temps

    Returns:
        Vitesse scalaire
    """
    if dt <= 0:
        return 0.0
    return calculate_distance(point1, point2) / dt


def point_in_polygon(
    point: Tuple[float, float],
    polygon: List[Tuple[float, float]]
) -> bool:
    """
    Vérifier si un point est à l'intérieur d'un polygone (Ray casting algorithm)

    Args:
        point: Point à tester (x, y)
        polygon: Liste des sommets du polygone

    Returns:
        True si le point est dans le polygone
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def line_intersection(
    line1: Tuple[Tuple[float, float], Tuple[float, float]],
    line2: Tuple[Tuple[float, float], Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    """
    Calculer l'intersection de deux lignes

    Args:
        line1: Première ligne ((x1, y1), (x2, y2))
        line2: Deuxième ligne ((x3, y3), (x4, y4))

    Returns:
        Point d'intersection ou None si parallèles
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Lignes parallèles

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)

    return (x, y)


def smooth_trajectory(
    points: List[Tuple[float, float]],
    sigma: float = 2.0
) -> List[Tuple[float, float]]:
    """
    Lisser une trajectoire avec un filtre gaussien

    Args:
        points: Liste de points (x, y)
        sigma: Écart-type du filtre gaussien

    Returns:
        Trajectoire lissée
    """
    if len(points) < 3:
        return points

    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])

    x_smooth = gaussian_filter1d(x_coords, sigma)
    y_smooth = gaussian_filter1d(y_coords, sigma)

    return list(zip(x_smooth, y_smooth))


def moving_average(
    values: List[float],
    window: int = 5
) -> List[float]:
    """
    Calculer la moyenne mobile

    Args:
        values: Liste de valeurs
        window: Taille de la fenêtre

    Returns:
        Liste des moyennes mobiles
    """
    if len(values) < window:
        return values

    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(np.mean(values[start:i + 1]))
    return result


def calculate_bbox_center(
    bbox: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    """
    Calculer le centre d'une bounding box

    Args:
        bbox: Bounding box (x1, y1, x2, y2)

    Returns:
        Centre (cx, cy)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_bbox_bottom_center(
    bbox: Tuple[float, float, float, float]
) -> Tuple[float, float]:
    """
    Calculer le centre bas d'une bounding box (position des pieds)

    Args:
        bbox: Bounding box (x1, y1, x2, y2)

    Returns:
        Centre bas (cx, y2)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def bbox_iou(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float]
) -> float:
    """
    Calculer l'IoU (Intersection over Union) de deux bounding boxes

    Args:
        bbox1: Première bbox (x1, y1, x2, y2)
        bbox2: Deuxième bbox (x1, y1, x2, y2)

    Returns:
        IoU (0.0 - 1.0)
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def scale_bbox(
    bbox: Tuple[float, float, float, float],
    scale: float,
    center: bool = True
) -> Tuple[float, float, float, float]:
    """
    Redimensionner une bounding box

    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        scale: Facteur d'échelle
        center: Si True, garder le centre fixe

    Returns:
        Bounding box redimensionnée
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    new_width = width * scale
    new_height = height * scale

    if center:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        return (
            cx - new_width / 2,
            cy - new_height / 2,
            cx + new_width / 2,
            cy + new_height / 2
        )
    else:
        return (x1, y1, x1 + new_width, y1 + new_height)


def clip_bbox(
    bbox: Tuple[float, float, float, float],
    width: int,
    height: int
) -> Tuple[int, int, int, int]:
    """
    Contraindre une bounding box aux dimensions de l'image

    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        width: Largeur de l'image
        height: Hauteur de l'image

    Returns:
        Bounding box contrainte (entiers)
    """
    x1 = max(0, min(int(bbox[0]), width - 1))
    y1 = max(0, min(int(bbox[1]), height - 1))
    x2 = max(0, min(int(bbox[2]), width))
    y2 = max(0, min(int(bbox[3]), height))
    return (x1, y1, x2, y2)


def rotate_point(
    point: Tuple[float, float],
    center: Tuple[float, float],
    angle_degrees: float
) -> Tuple[float, float]:
    """
    Faire pivoter un point autour d'un centre

    Args:
        point: Point à pivoter (x, y)
        center: Centre de rotation (cx, cy)
        angle_degrees: Angle de rotation en degrés

    Returns:
        Point pivoté
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    x, y = point
    cx, cy = center

    # Translation vers l'origine
    x -= cx
    y -= cy

    # Rotation
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a

    # Translation inverse
    return (x_new + cx, y_new + cy)


def calculate_polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """
    Calculer l'aire d'un polygone (formule du lacet)

    Args:
        polygon: Liste des sommets du polygone

    Returns:
        Aire du polygone
    """
    n = len(polygon)
    if n < 3:
        return 0.0

    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2.0


def convex_hull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Calculer l'enveloppe convexe d'un ensemble de points

    Args:
        points: Liste de points

    Returns:
        Points de l'enveloppe convexe
    """
    import cv2
    if len(points) < 3:
        return points

    points_array = np.array(points, dtype=np.float32)
    hull = cv2.convexHull(points_array)
    return [tuple(p[0]) for p in hull]
