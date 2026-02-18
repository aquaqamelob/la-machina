"""
Football Analysis - Dessin des trajectoires
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils.colors import ColorPalette, interpolate_color, get_team_color
from utils.geometry import smooth_trajectory


@dataclass
class TrajectoryStyle:
    """Style de trajectoire"""
    color: Tuple[int, int, int] = (255, 255, 255)
    thickness: int = 2
    fade: bool = True
    fade_start: float = 0.3  # Opacité au début de la trajectoire
    fade_end: float = 1.0    # Opacité à la fin
    dotted: bool = False
    dot_spacing: int = 10
    arrow: bool = False
    smooth: bool = True
    smooth_sigma: float = 2.0


class TrajectoryDrawer:
    """
    Dessinateur de trajectoires pour joueurs et ballon

    Crée des visualisations élégantes des mouvements.
    """

    def __init__(self):
        """Initialiser le drawer"""
        self.default_player_style = TrajectoryStyle(
            color=ColorPalette.TEXT_WHITE,
            thickness=2,
            fade=True,
            smooth=True
        )

        self.default_ball_style = TrajectoryStyle(
            color=ColorPalette.BALL,
            thickness=3,
            fade=True,
            dotted=True,
            dot_spacing=8,
            smooth=True,
            smooth_sigma=1.5
        )

    def draw_trajectory(
        self,
        frame: np.ndarray,
        positions: List[Tuple[float, float]],
        style: Optional[TrajectoryStyle] = None,
        team: Optional[str] = None
    ) -> np.ndarray:
        """
        Dessiner une trajectoire sur un frame

        Args:
            frame: Image BGR
            positions: Liste des positions (x, y)
            style: Style de la trajectoire
            team: Équipe pour la couleur

        Returns:
            Image avec trajectoire
        """
        if len(positions) < 2:
            return frame

        result = frame.copy()
        style = style or self.default_player_style

        # Déterminer la couleur
        if team:
            color = get_team_color(team)
        else:
            color = style.color

        # Lisser la trajectoire si demandé
        if style.smooth and len(positions) >= 3:
            positions = smooth_trajectory(positions, style.smooth_sigma)

        # Convertir en points entiers
        points = [(int(x), int(y)) for x, y in positions]

        if style.dotted:
            # Trajectoire en pointillés
            result = self._draw_dotted_trajectory(
                result, points, color, style
            )
        else:
            # Trajectoire continue
            result = self._draw_continuous_trajectory(
                result, points, color, style
            )

        # Ajouter une flèche de direction si demandé
        if style.arrow and len(points) >= 2:
            result = self._draw_direction_arrow(result, points[-2:], color, style)

        return result

    def _draw_continuous_trajectory(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        style: TrajectoryStyle
    ) -> np.ndarray:
        """Dessiner une trajectoire continue avec fade"""
        result = frame.copy()

        for i in range(len(points) - 1):
            # Calculer l'opacité
            if style.fade:
                progress = i / (len(points) - 1)
                alpha = style.fade_start + (style.fade_end - style.fade_start) * progress
            else:
                alpha = 1.0

            # Créer une version avec alpha de la couleur
            pt1 = points[i]
            pt2 = points[i + 1]

            # Dessiner sur un overlay
            overlay = result.copy()
            cv2.line(overlay, pt1, pt2, color, style.thickness, cv2.LINE_AA)
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        return result

    def _draw_dotted_trajectory(
        self,
        frame: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        style: TrajectoryStyle
    ) -> np.ndarray:
        """Dessiner une trajectoire en pointillés"""
        result = frame.copy()

        # Calculer la distance totale
        total_dist = 0
        distances = [0]
        for i in range(1, len(points)):
            d = np.sqrt(
                (points[i][0] - points[i-1][0])**2 +
                (points[i][1] - points[i-1][1])**2
            )
            total_dist += d
            distances.append(total_dist)

        # Dessiner les points le long de la trajectoire
        current_dist = 0
        point_idx = 0

        while current_dist < total_dist:
            # Trouver la position sur la trajectoire
            while point_idx < len(distances) - 1 and distances[point_idx + 1] < current_dist:
                point_idx += 1

            if point_idx >= len(points) - 1:
                break

            # Interpoler la position
            if distances[point_idx + 1] - distances[point_idx] > 0:
                t = (current_dist - distances[point_idx]) / (
                    distances[point_idx + 1] - distances[point_idx]
                )
            else:
                t = 0

            x = int(points[point_idx][0] + t * (points[point_idx + 1][0] - points[point_idx][0]))
            y = int(points[point_idx][1] + t * (points[point_idx + 1][1] - points[point_idx][1]))

            # Calculer l'opacité
            if style.fade:
                progress = current_dist / total_dist if total_dist > 0 else 1.0
                alpha = style.fade_start + (style.fade_end - style.fade_start) * progress
            else:
                alpha = 1.0

            # Dessiner le point
            overlay = result.copy()
            cv2.circle(overlay, (x, y), style.thickness, color, -1, cv2.LINE_AA)
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

            current_dist += style.dot_spacing

        return result

    def _draw_direction_arrow(
        self,
        frame: np.ndarray,
        last_two_points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        style: TrajectoryStyle
    ) -> np.ndarray:
        """Dessiner une flèche de direction"""
        result = frame.copy()
        pt1, pt2 = last_two_points

        # Vecteur de direction
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        length = np.sqrt(dx**2 + dy**2)

        if length < 5:
            return result

        # Normaliser
        dx /= length
        dy /= length

        # Taille de la flèche
        arrow_size = 15

        # Points de la flèche
        tip = pt2
        left = (
            int(pt2[0] - arrow_size * (dx + 0.5 * dy)),
            int(pt2[1] - arrow_size * (dy - 0.5 * dx))
        )
        right = (
            int(pt2[0] - arrow_size * (dx - 0.5 * dy)),
            int(pt2[1] - arrow_size * (dy + 0.5 * dx))
        )

        # Dessiner la flèche
        cv2.fillPoly(result, [np.array([tip, left, right])], color, cv2.LINE_AA)

        return result

    def draw_player_trajectory(
        self,
        frame: np.ndarray,
        positions: List[Tuple[float, float]],
        player_id: int,
        team: Optional[str] = None,
        max_length: int = None
    ) -> np.ndarray:
        """
        Dessiner la trajectoire d'un joueur

        Args:
            frame: Image BGR
            positions: Historique des positions
            player_id: ID du joueur
            team: Équipe
            max_length: Longueur max de la trajectoire à afficher

        Returns:
            Image avec trajectoire
        """
        max_length = max_length or config.visualization.trajectory_length

        # Tronquer l'historique
        if len(positions) > max_length:
            positions = positions[-max_length:]

        style = TrajectoryStyle(
            color=get_team_color(team) if team else ColorPalette.TEXT_WHITE,
            thickness=2,
            fade=config.visualization.trajectory_fade,
            smooth=True
        )

        return self.draw_trajectory(frame, positions, style, team)

    def draw_ball_trajectory(
        self,
        frame: np.ndarray,
        positions: List[Tuple[float, float]],
        max_length: int = None
    ) -> np.ndarray:
        """
        Dessiner la trajectoire du ballon

        Args:
            frame: Image BGR
            positions: Historique des positions du ballon
            max_length: Longueur max à afficher

        Returns:
            Image avec trajectoire
        """
        max_length = max_length or config.visualization.trajectory_length

        if len(positions) > max_length:
            positions = positions[-max_length:]

        return self.draw_trajectory(frame, positions, self.default_ball_style)

    def draw_all_trajectories(
        self,
        frame: np.ndarray,
        player_trajectories: Dict[int, List[Tuple[float, float]]],
        player_teams: Dict[int, str] = None,
        ball_trajectory: List[Tuple[float, float]] = None,
        max_length: int = None
    ) -> np.ndarray:
        """
        Dessiner toutes les trajectoires

        Args:
            frame: Image BGR
            player_trajectories: Trajectoires par joueur
            player_teams: Équipes par joueur
            ball_trajectory: Trajectoire du ballon
            max_length: Longueur max des trajectoires

        Returns:
            Image avec toutes les trajectoires
        """
        result = frame.copy()
        player_teams = player_teams or {}

        # Dessiner les trajectoires des joueurs
        for player_id, positions in player_trajectories.items():
            team = player_teams.get(player_id)
            result = self.draw_player_trajectory(
                result, positions, player_id, team, max_length
            )

        # Dessiner la trajectoire du ballon
        if ball_trajectory:
            result = self.draw_ball_trajectory(result, ball_trajectory, max_length)

        return result

    def draw_movement_vector(
        self,
        frame: np.ndarray,
        position: Tuple[float, float],
        velocity: Tuple[float, float],
        color: Tuple[int, int, int] = None,
        scale: float = 5.0
    ) -> np.ndarray:
        """
        Dessiner un vecteur de mouvement

        Args:
            frame: Image BGR
            position: Position actuelle
            velocity: Vecteur vitesse (vx, vy)
            color: Couleur de la flèche
            scale: Échelle du vecteur

        Returns:
            Image avec vecteur
        """
        result = frame.copy()
        color = color or ColorPalette.TEXT_WHITE

        pt1 = (int(position[0]), int(position[1]))
        pt2 = (
            int(position[0] + velocity[0] * scale),
            int(position[1] + velocity[1] * scale)
        )

        cv2.arrowedLine(
            result, pt1, pt2, color, 2,
            cv2.LINE_AA, tipLength=0.3
        )

        return result

    def create_trajectory_overlay(
        self,
        width: int,
        height: int,
        trajectories: Dict[int, List[Tuple[float, float]]],
        teams: Dict[int, str] = None,
        alpha: float = 0.7
    ) -> np.ndarray:
        """
        Créer un overlay transparent avec toutes les trajectoires

        Args:
            width: Largeur de l'overlay
            height: Hauteur de l'overlay
            trajectories: Trajectoires par joueur
            teams: Équipes par joueur
            alpha: Transparence de l'overlay

        Returns:
            Image BGRA de l'overlay
        """
        # Créer un fond transparent
        overlay = np.zeros((height, width, 4), dtype=np.uint8)
        teams = teams or {}

        # Dessiner sur un frame BGR temporaire
        temp = np.zeros((height, width, 3), dtype=np.uint8)

        for player_id, positions in trajectories.items():
            team = teams.get(player_id)
            temp = self.draw_player_trajectory(
                temp, positions, player_id, team
            )

        # Copier vers l'overlay avec alpha
        overlay[:, :, :3] = temp
        overlay[:, :, 3] = (np.any(temp > 0, axis=2) * 255 * alpha).astype(np.uint8)

        return overlay
