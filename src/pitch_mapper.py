"""
Football Analysis - Homographie et mapping du terrain
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils.colors import ColorPalette


@dataclass
class PitchPoints:
    """Points de référence sur le terrain"""
    # Points en pixels (vidéo)
    video_points: List[Tuple[float, float]]
    # Points correspondants en mètres (terrain réel)
    pitch_points: List[Tuple[float, float]]


class PitchMapper:
    """
    Mapper pour la conversion entre vue vidéo et vue 2D du terrain

    Utilise l'homographie pour transformer les coordonnées.
    """

    def __init__(
        self,
        pitch_length: float = None,
        pitch_width: float = None,
        viz_width: int = None,
        viz_height: int = None
    ):
        """
        Initialiser le mapper

        Args:
            pitch_length: Longueur du terrain en mètres
            pitch_width: Largeur du terrain en mètres
            viz_width: Largeur de la visualisation 2D en pixels
            viz_height: Hauteur de la visualisation 2D en pixels
        """
        self.pitch_length = pitch_length or config.pitch.length
        self.pitch_width = pitch_width or config.pitch.width
        self.viz_width = viz_width or config.pitch.viz_width
        self.viz_height = viz_height or config.pitch.viz_height

        # Matrice d'homographie
        self.homography_matrix: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None

        # Points de calibration
        self.calibration_points: Optional[PitchPoints] = None

        # Facteurs d'échelle pour la visualisation
        self.scale_x = self.viz_width / self.pitch_length
        self.scale_y = self.viz_height / self.pitch_width

    def calibrate(
        self,
        video_points: List[Tuple[float, float]],
        pitch_points: List[Tuple[float, float]]
    ) -> bool:
        """
        Calibrer l'homographie avec des points de correspondance

        Args:
            video_points: Points dans la vidéo (pixels)
            pitch_points: Points sur le terrain (mètres)

        Returns:
            True si calibration réussie
        """
        if len(video_points) < 4 or len(pitch_points) < 4:
            return False

        # Convertir en arrays numpy
        src = np.array(video_points, dtype=np.float32)
        dst = np.array(pitch_points, dtype=np.float32)

        # Calculer l'homographie
        self.homography_matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC)

        if self.homography_matrix is None:
            return False

        # Calculer l'inverse
        self.inverse_homography = np.linalg.inv(self.homography_matrix)

        # Stocker les points de calibration
        self.calibration_points = PitchPoints(
            video_points=video_points,
            pitch_points=pitch_points
        )

        return True

    def calibrate_from_corners(
        self,
        corners: List[Tuple[float, float]]
    ) -> bool:
        """
        Calibrer à partir des 4 coins du terrain visibles

        Args:
            corners: 4 coins dans l'ordre: haut-gauche, haut-droite, bas-droite, bas-gauche

        Returns:
            True si calibration réussie
        """
        if len(corners) != 4:
            return False

        # Points correspondants sur le terrain réel (en mètres)
        pitch_corners = [
            (0, 0),                             # Haut-gauche
            (self.pitch_length, 0),             # Haut-droite
            (self.pitch_length, self.pitch_width),  # Bas-droite
            (0, self.pitch_width)               # Bas-gauche
        ]

        return self.calibrate(corners, pitch_corners)

    def transform_point(
        self,
        point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transformer un point vidéo en coordonnées terrain

        Args:
            point: Point en pixels (x, y)

        Returns:
            Point en mètres (x, y) ou None
        """
        if self.homography_matrix is None:
            return None

        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)

        return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))

    def transform_to_video(
        self,
        point: Tuple[float, float]
    ) -> Optional[Tuple[float, float]]:
        """
        Transformer un point terrain en coordonnées vidéo

        Args:
            point: Point en mètres (x, y)

        Returns:
            Point en pixels (x, y) ou None
        """
        if self.inverse_homography is None:
            return None

        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_homography)

        return (float(transformed[0, 0, 0]), float(transformed[0, 0, 1]))

    def point_to_viz(
        self,
        point: Tuple[float, float]
    ) -> Tuple[int, int]:
        """
        Convertir un point terrain (mètres) en coordonnées de visualisation

        Args:
            point: Point en mètres

        Returns:
            Point en pixels pour la visualisation
        """
        x = int(point[0] * self.scale_x)
        y = int(point[1] * self.scale_y)
        return (x, y)

    def video_to_viz(
        self,
        point: Tuple[float, float]
    ) -> Optional[Tuple[int, int]]:
        """
        Convertir directement un point vidéo en coordonnées de visualisation

        Args:
            point: Point vidéo en pixels

        Returns:
            Point pour la visualisation ou None
        """
        pitch_point = self.transform_point(point)
        if pitch_point is None:
            return None
        return self.point_to_viz(pitch_point)

    def create_pitch_image(
        self,
        background_color: Tuple[int, int, int] = None,
        line_color: Tuple[int, int, int] = None,
        line_thickness: int = 2
    ) -> np.ndarray:
        """
        Créer une image 2D du terrain de football

        Args:
            background_color: Couleur de fond
            line_color: Couleur des lignes
            line_thickness: Épaisseur des lignes

        Returns:
            Image BGR du terrain
        """
        background_color = background_color or (34, 139, 34)  # Vert terrain
        line_color = line_color or (255, 255, 255)

        # Créer l'image
        pitch = np.zeros((self.viz_height, self.viz_width, 3), dtype=np.uint8)
        pitch[:] = background_color

        # Convertir les dimensions en pixels
        def m_to_px(m_x, m_y):
            return (int(m_x * self.scale_x), int(m_y * self.scale_y))

        # Lignes extérieures
        cv2.rectangle(
            pitch,
            m_to_px(0, 0),
            m_to_px(self.pitch_length, self.pitch_width),
            line_color, line_thickness
        )

        # Ligne médiane
        center_x = self.pitch_length / 2
        cv2.line(
            pitch,
            m_to_px(center_x, 0),
            m_to_px(center_x, self.pitch_width),
            line_color, line_thickness
        )

        # Cercle central
        center = m_to_px(center_x, self.pitch_width / 2)
        radius = int(9.15 * self.scale_x)  # 9.15m de rayon
        cv2.circle(pitch, center, radius, line_color, line_thickness)

        # Point central
        cv2.circle(pitch, center, 5, line_color, -1)

        # Surface de réparation gauche
        penalty_length = config.pitch.penalty_area_length
        penalty_width = config.pitch.penalty_area_width
        penalty_top = (self.pitch_width - penalty_width) / 2

        cv2.rectangle(
            pitch,
            m_to_px(0, penalty_top),
            m_to_px(penalty_length, penalty_top + penalty_width),
            line_color, line_thickness
        )

        # Surface de réparation droite
        cv2.rectangle(
            pitch,
            m_to_px(self.pitch_length - penalty_length, penalty_top),
            m_to_px(self.pitch_length, penalty_top + penalty_width),
            line_color, line_thickness
        )

        # Surface de but gauche
        goal_length = config.pitch.goal_area_length
        goal_width = config.pitch.goal_area_width
        goal_top = (self.pitch_width - goal_width) / 2

        cv2.rectangle(
            pitch,
            m_to_px(0, goal_top),
            m_to_px(goal_length, goal_top + goal_width),
            line_color, line_thickness
        )

        # Surface de but droite
        cv2.rectangle(
            pitch,
            m_to_px(self.pitch_length - goal_length, goal_top),
            m_to_px(self.pitch_length, goal_top + goal_width),
            line_color, line_thickness
        )

        # Points de penalty
        penalty_spot = config.pitch.penalty_spot_distance
        cv2.circle(
            pitch,
            m_to_px(penalty_spot, self.pitch_width / 2),
            5, line_color, -1
        )
        cv2.circle(
            pitch,
            m_to_px(self.pitch_length - penalty_spot, self.pitch_width / 2),
            5, line_color, -1
        )

        # Arcs de réparation
        arc_center_left = m_to_px(penalty_spot, self.pitch_width / 2)
        arc_center_right = m_to_px(self.pitch_length - penalty_spot, self.pitch_width / 2)
        arc_radius = int(9.15 * self.scale_x)

        # Arc gauche (portion visible hors surface)
        cv2.ellipse(
            pitch, arc_center_left, (arc_radius, arc_radius),
            0, -53, 53, line_color, line_thickness
        )

        # Arc droit
        cv2.ellipse(
            pitch, arc_center_right, (arc_radius, arc_radius),
            0, 127, 233, line_color, line_thickness
        )

        # Coins
        corner_radius = int(1.0 * self.scale_x)  # 1m de rayon
        corners = [
            (m_to_px(0, 0), 0, 90),
            (m_to_px(self.pitch_length, 0), 90, 180),
            (m_to_px(self.pitch_length, self.pitch_width), 180, 270),
            (m_to_px(0, self.pitch_width), 270, 360)
        ]
        for center, start, end in corners:
            cv2.ellipse(
                pitch, center, (corner_radius, corner_radius),
                0, start, end, line_color, line_thickness
            )

        return pitch

    def draw_players_on_pitch(
        self,
        pitch_image: np.ndarray,
        player_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, str] = None,
        ball_position: Tuple[float, float] = None,
        from_video: bool = True
    ) -> np.ndarray:
        """
        Dessiner les joueurs sur l'image du terrain

        Args:
            pitch_image: Image du terrain 2D
            player_positions: Positions des joueurs (pixels vidéo ou mètres)
            player_teams: Équipes des joueurs
            ball_position: Position du ballon
            from_video: True si les positions sont en pixels vidéo

        Returns:
            Image avec les joueurs
        """
        result = pitch_image.copy()
        player_teams = player_teams or {}

        for player_id, pos in player_positions.items():
            # Convertir en coordonnées de visualisation
            if from_video:
                viz_pos = self.video_to_viz(pos)
            else:
                viz_pos = self.point_to_viz(pos)

            if viz_pos is None:
                continue

            # Déterminer la couleur
            team = player_teams.get(player_id, "unknown")
            if team == "team_a":
                color = ColorPalette.TEAM_A
            elif team == "team_b":
                color = ColorPalette.TEAM_B
            elif team == "referee":
                color = ColorPalette.REFEREE
            else:
                color = ColorPalette.UNKNOWN

            # Dessiner le joueur
            cv2.circle(result, viz_pos, 8, color, -1, cv2.LINE_AA)
            cv2.circle(result, viz_pos, 8, (255, 255, 255), 1, cv2.LINE_AA)

        # Dessiner le ballon
        if ball_position:
            if from_video:
                ball_viz = self.video_to_viz(ball_position)
            else:
                ball_viz = self.point_to_viz(ball_position)

            if ball_viz:
                cv2.circle(result, ball_viz, 6, ColorPalette.BALL, -1, cv2.LINE_AA)
                cv2.circle(result, ball_viz, 6, (0, 0, 0), 1, cv2.LINE_AA)

        return result

    def create_bird_eye_view(
        self,
        frame: np.ndarray,
        output_size: Tuple[int, int] = None
    ) -> Optional[np.ndarray]:
        """
        Créer une vue aérienne du terrain depuis la vidéo

        Args:
            frame: Image vidéo
            output_size: Taille de sortie (width, height)

        Returns:
            Vue aérienne ou None si pas calibré
        """
        if self.homography_matrix is None:
            return None

        output_size = output_size or (self.viz_width, self.viz_height)

        # Transformer l'image
        warped = cv2.warpPerspective(
            frame,
            self.homography_matrix,
            output_size
        )

        return warped

    def get_pitch_zone(
        self,
        position: Tuple[float, float]
    ) -> str:
        """
        Déterminer la zone du terrain pour une position

        Args:
            position: Position en mètres

        Returns:
            Nom de la zone ("left_penalty", "left_half", "center", "right_half", "right_penalty")
        """
        x, y = position

        penalty_length = config.pitch.penalty_area_length
        penalty_top = (self.pitch_width - config.pitch.penalty_area_width) / 2
        penalty_bottom = penalty_top + config.pitch.penalty_area_width

        # Surface de réparation gauche
        if x <= penalty_length and penalty_top <= y <= penalty_bottom:
            return "left_penalty"

        # Surface de réparation droite
        if x >= self.pitch_length - penalty_length and penalty_top <= y <= penalty_bottom:
            return "right_penalty"

        # Moitié gauche
        if x < self.pitch_length / 2:
            return "left_half"

        # Moitié droite
        return "right_half"

    def is_on_pitch(self, position: Tuple[float, float]) -> bool:
        """Vérifier si une position est sur le terrain"""
        x, y = position
        return 0 <= x <= self.pitch_length and 0 <= y <= self.pitch_width

    def estimate_homography_from_lines(
        self,
        frame: np.ndarray
    ) -> bool:
        """
        Estimer l'homographie automatiquement depuis les lignes du terrain

        Args:
            frame: Image vidéo

        Returns:
            True si réussi
        """
        # Détection de lignes avec Hough
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

        if lines is None or len(lines) < 4:
            return False

        # Cette fonction nécessiterait une implémentation plus sophistiquée
        # pour identifier automatiquement les lignes du terrain
        # Pour l'instant, on retourne False et on utilise la calibration manuelle
        return False

    def reset(self):
        """Réinitialiser la calibration"""
        self.homography_matrix = None
        self.inverse_homography = None
        self.calibration_points = None
