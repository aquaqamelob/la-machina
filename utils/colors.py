"""
Football Analysis - Utilitaires de couleurs
"""

from typing import Tuple, List
import numpy as np
import cv2


class ColorPalette:
    """Palette de couleurs pour la visualisation"""

    # Couleurs principales (BGR pour OpenCV)
    TEAM_A = (219, 152, 52)       # #3498db bleu
    TEAM_B = (60, 76, 231)        # #e74c3c rouge
    REFEREE = (0, 165, 255)       # #ffa500 orange (arbitre)
    GOALKEEPER_A = (255, 191, 0)  # #00bfff cyan
    GOALKEEPER_B = (128, 0, 128)  # violet
    BALL = (0, 196, 241)          # #f1c40f jaune
    UNKNOWN = (128, 128, 128)     # gris

    # Couleurs pour l'overlay
    OVERLAY_BG = (0, 0, 0)
    TEXT_WHITE = (255, 255, 255)
    TEXT_GRAY = (200, 200, 200)

    # Couleurs pour les trajectoires
    TRAJECTORY_START = (0, 255, 255)   # Jaune
    TRAJECTORY_END = (0, 165, 255)     # Orange

    # Couleurs pour le terrain 2D
    PITCH_GREEN = (34, 139, 34)        # Vert terrain
    PITCH_LINES = (255, 255, 255)      # Lignes blanches

    # Palette étendue pour plusieurs joueurs
    PLAYER_COLORS = [
        (255, 0, 0),     # Rouge
        (0, 255, 0),     # Vert
        (0, 0, 255),     # Bleu
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Jaune
        (128, 0, 0),     # Marron
        (0, 128, 0),     # Vert foncé
        (0, 0, 128),     # Bleu marine
        (128, 128, 0),   # Olive
        (128, 0, 128),   # Violet
        (0, 128, 128),   # Teal
    ]

    @classmethod
    def get_player_color(cls, player_id: int) -> Tuple[int, int, int]:
        """Obtenir une couleur unique pour un joueur basée sur son ID"""
        return cls.PLAYER_COLORS[player_id % len(cls.PLAYER_COLORS)]


def get_team_color(team: str) -> Tuple[int, int, int]:
    """
    Obtenir la couleur d'une équipe

    Args:
        team: "team_a", "team_b", "referee", "goalkeeper_a", "goalkeeper_b", ou "unknown"

    Returns:
        Couleur BGR
    """
    colors = {
        "team_a": ColorPalette.TEAM_A,
        "team_b": ColorPalette.TEAM_B,
        "referee": ColorPalette.REFEREE,
        "goalkeeper_a": ColorPalette.GOALKEEPER_A,
        "goalkeeper_b": ColorPalette.GOALKEEPER_B,
        "ball": ColorPalette.BALL,
        "unknown": ColorPalette.UNKNOWN
    }
    return colors.get(team, ColorPalette.UNKNOWN)


def interpolate_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    factor: float
) -> Tuple[int, int, int]:
    """
    Interpoler entre deux couleurs

    Args:
        color1: Première couleur (BGR)
        color2: Deuxième couleur (BGR)
        factor: Facteur d'interpolation (0.0 = color1, 1.0 = color2)

    Returns:
        Couleur interpolée
    """
    factor = max(0.0, min(1.0, factor))
    return tuple(
        int(c1 + (c2 - c1) * factor)
        for c1, c2 in zip(color1, color2)
    )


def bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convertir BGR vers RGB"""
    return (color[2], color[1], color[0])


def rgb_to_bgr(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Convertir RGB vers BGR"""
    return (color[2], color[1], color[0])


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """
    Convertir une couleur hexadécimale vers BGR

    Args:
        hex_color: Couleur au format "#RRGGBB" ou "RRGGBB"

    Returns:
        Couleur BGR
    """
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])


def bgr_to_hex(color: Tuple[int, int, int]) -> str:
    """
    Convertir une couleur BGR vers hexadécimal

    Args:
        color: Couleur BGR

    Returns:
        Couleur au format "#RRGGBB"
    """
    return f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"


def apply_alpha(
    background: np.ndarray,
    overlay: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    Appliquer un overlay avec transparence

    Args:
        background: Image de fond
        overlay: Image à superposer
        alpha: Transparence (0.0 = transparent, 1.0 = opaque)

    Returns:
        Image fusionnée
    """
    return cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0)


def create_gradient(
    width: int,
    height: int,
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    direction: str = "horizontal"
) -> np.ndarray:
    """
    Créer un gradient de couleur

    Args:
        width: Largeur de l'image
        height: Hauteur de l'image
        color1: Couleur de départ (BGR)
        color2: Couleur de fin (BGR)
        direction: "horizontal" ou "vertical"

    Returns:
        Image avec gradient
    """
    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    if direction == "horizontal":
        for x in range(width):
            factor = x / (width - 1) if width > 1 else 0
            color = interpolate_color(color1, color2, factor)
            gradient[:, x] = color
    else:  # vertical
        for y in range(height):
            factor = y / (height - 1) if height > 1 else 0
            color = interpolate_color(color1, color2, factor)
            gradient[y, :] = color

    return gradient


def get_dominant_color(
    image: np.ndarray,
    mask: np.ndarray = None,
    k: int = 1
) -> List[Tuple[int, int, int]]:
    """
    Extraire les couleurs dominantes d'une image

    Args:
        image: Image BGR
        mask: Masque optionnel (pixels blancs = inclus)
        k: Nombre de couleurs à extraire

    Returns:
        Liste des couleurs dominantes (BGR)
    """
    from sklearn.cluster import KMeans

    # Appliquer le masque si fourni
    if mask is not None:
        pixels = image[mask > 0].reshape(-1, 3)
    else:
        pixels = image.reshape(-1, 3)

    if len(pixels) < k:
        return [(0, 0, 0)] * k

    # Clustering K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)

    # Récupérer les centres des clusters
    colors = kmeans.cluster_centers_.astype(int)

    # Trier par fréquence
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    sorted_indices = np.argsort(-counts)

    return [tuple(colors[i]) for i in sorted_indices]


def color_distance(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int]
) -> float:
    """
    Calculer la distance euclidienne entre deux couleurs

    Args:
        color1: Première couleur (BGR)
        color2: Deuxième couleur (BGR)

    Returns:
        Distance entre les couleurs
    """
    return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))


def is_similar_color(
    color1: Tuple[int, int, int],
    color2: Tuple[int, int, int],
    threshold: float = 50.0
) -> bool:
    """
    Vérifier si deux couleurs sont similaires

    Args:
        color1: Première couleur (BGR)
        color2: Deuxième couleur (BGR)
        threshold: Seuil de distance maximum

    Returns:
        True si les couleurs sont similaires
    """
    return color_distance(color1, color2) < threshold
