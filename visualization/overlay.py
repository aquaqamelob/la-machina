"""
Football Analysis - Overlay vidéo et annotations
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, SKELETON_CONNECTIONS, COCO_KEYPOINTS
from utils.colors import ColorPalette, get_team_color, interpolate_color
from src.tracker import TrackedObject
from src.pose_estimator import Pose


class VideoOverlay:
    """
    Classe pour les annotations visuelles sur la vidéo

    Style professionnel broadcast TV.
    """

    def __init__(self):
        """Initialiser l'overlay"""
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = config.visualization.font_scale
        self.font_thickness = config.visualization.font_thickness

        # Cache pour le terrain 2D minimap
        self._minimap_cache: Optional[np.ndarray] = None

    def draw_player_circle(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        team: str = "unknown"
    ) -> np.ndarray:
        """
        Dessiner un cercle/ellipse de couleur sous un joueur

        Args:
            frame: Image BGR
            bbox: Bounding box (x1, y1, x2, y2)
            team: Équipe du joueur (orange = arbitre)

        Returns:
            Image annotée
        """
        result = frame.copy()
        x1, y1, x2, y2 = map(int, bbox)

        # Centre bas de la bbox (position des pieds)
        cx = (x1 + x2) // 2
        cy = y2

        # Couleur selon l'équipe
        color = get_team_color(team)

        # Taille de l'ellipse basée sur la largeur de la bbox
        width = x2 - x1
        axes = (int(width * 0.45), int(width * 0.18))

        # Ellipse avec transparence
        overlay = result.copy()
        cv2.ellipse(overlay, (cx, cy), axes, 0, 0, 360, color, -1, cv2.LINE_AA)
        result = cv2.addWeighted(overlay, 0.6, result, 0.4, 0)

        # Contour de l'ellipse
        cv2.ellipse(result, (cx, cy), axes, 0, 0, 360, color, 2, cv2.LINE_AA)

        return result

    def draw_player_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        team: str = "unknown",
        player_id: Optional[int] = None,
        speed: Optional[float] = None,
        confidence: float = 1.0
    ) -> np.ndarray:
        """Redirige vers draw_player_circle (version simplifiée)"""
        return self.draw_player_circle(frame, bbox, team)

    def draw_formation_lines(
        self,
        frame: np.ndarray,
        players: List[TrackedObject],
        n_lines: int = 4
    ) -> np.ndarray:
        """
        Dessiner les lignes de formation tactiques

        Dessine des liens quand 3+ joueurs sont alignés horizontalement.
        Chaque joueur a maximum 2 connexions (voisins gauche/droite).

        Args:
            frame: Image BGR
            players: Liste des joueurs trackés
            n_lines: Non utilisé (gardé pour compatibilité)

        Returns:
            Image annotée
        """
        result = frame.copy()

        # Séparer les joueurs par équipe (exclure arbitres et unknown)
        teams = defaultdict(list)
        for player in players:
            if player.team in ["team_a", "team_b"]:
                teams[player.team].append(player)

        # Pour chaque équipe
        for team, team_players in teams.items():
            if len(team_players) < 3:
                continue

            # Obtenir les positions (centre bas = pieds)
            positions = []
            for p in team_players:
                x1, y1, x2, y2 = p.bbox
                cx = (x1 + x2) / 2
                cy = y2  # bas de la bbox
                positions.append((cx, cy))

            team_color = get_team_color(team)

            # Trouver les lignes de 3+ joueurs alignés horizontalement
            lines = self._find_horizontal_lines(positions, min_players=3, y_tolerance=80)

            # Dessiner les lignes (chaque joueur connecté à max 2 voisins)
            for line in lines:
                # Trier par X (gauche à droite)
                line.sort(key=lambda p: p[0])

                # Relier les joueurs adjacents (max 2 connexions par joueur)
                for i in range(len(line) - 1):
                    pt1 = (int(line[i][0]), int(line[i][1]))
                    pt2 = (int(line[i + 1][0]), int(line[i + 1][1]))
                    overlay = result.copy()
                    cv2.line(overlay, pt1, pt2, team_color, 2, cv2.LINE_AA)
                    result = cv2.addWeighted(overlay, 0.6, result, 0.4, 0)

        return result

    def _find_horizontal_lines(
        self,
        positions: List[Tuple[float, float]],
        min_players: int = 3,
        y_tolerance: float = 80
    ) -> List[List[Tuple[float, float]]]:
        """
        Trouver les groupes de joueurs alignés horizontalement

        Args:
            positions: Liste de positions (x, y)
            min_players: Nombre minimum de joueurs pour former une ligne
            y_tolerance: Tolérance en pixels pour considérer deux joueurs sur la même ligne

        Returns:
            Liste de lignes, chaque ligne contenant les positions des joueurs
        """
        if len(positions) < min_players:
            return []

        lines = []
        used = set()

        # Trier par Y pour traiter les lignes du haut vers le bas
        indexed_positions = [(i, pos) for i, pos in enumerate(positions)]
        indexed_positions.sort(key=lambda x: x[1][1])

        # Pour chaque position non utilisée, chercher les autres sur la même ligne Y
        for i, (idx1, (x1, y1)) in enumerate(indexed_positions):
            if idx1 in used:
                continue

            line = [(x1, y1)]
            line_indices = {idx1}

            for j, (idx2, (x2, y2)) in enumerate(indexed_positions):
                if idx2 in used or idx2 == idx1:
                    continue

                # Même ligne horizontale si Y proche
                if abs(y2 - y1) < y_tolerance:
                    line.append((x2, y2))
                    line_indices.add(idx2)

            # Si on a assez de joueurs alignés
            if len(line) >= min_players:
                lines.append(line)
                used.update(line_indices)

        return lines

    def draw_pose_skeleton(
        self,
        frame: np.ndarray,
        pose: Pose,
        team: str = "unknown",
        style: str = "elegant"
    ) -> np.ndarray:
        """
        Dessiner le squelette d'un joueur

        Args:
            frame: Image BGR
            pose: Pose du joueur
            team: Équipe pour la couleur
            style: Style de dessin ("elegant", "simple", "detailed")

        Returns:
            Image annotée
        """
        result = frame.copy()
        color = get_team_color(team)
        thickness = config.visualization.skeleton_thickness

        # Dessiner les connexions
        for start_name, end_name in SKELETON_CONNECTIONS:
            start_pt = pose.get_point(start_name)
            end_pt = pose.get_point(end_name)

            if start_pt and end_pt:
                pt1 = (int(start_pt[0]), int(start_pt[1]))
                pt2 = (int(end_pt[0]), int(end_pt[1]))

                if style == "elegant":
                    # Ligne avec dégradé
                    cv2.line(result, pt1, pt2, color, thickness, cv2.LINE_AA)
                elif style == "detailed":
                    # Ligne plus épaisse avec contour
                    cv2.line(result, pt1, pt2, (255, 255, 255), thickness + 2, cv2.LINE_AA)
                    cv2.line(result, pt1, pt2, color, thickness, cv2.LINE_AA)
                else:  # simple
                    cv2.line(result, pt1, pt2, color, thickness)

        # Dessiner les points d'articulation
        for name, kp in pose.keypoints.items():
            if kp.is_visible:
                pt = (int(kp.x), int(kp.y))

                if style == "elegant":
                    cv2.circle(result, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(result, pt, 3, color, -1, cv2.LINE_AA)
                elif style == "detailed":
                    cv2.circle(result, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(result, pt, 4, color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(result, pt, 3, color, -1)

        return result

    def draw_ball(
        self,
        frame: np.ndarray,
        position: Tuple[float, float],
        trajectory: List[Tuple[float, float]] = None,
        confidence: float = 1.0
    ) -> np.ndarray:
        """
        Dessiner le ballon avec effet visuel et traînée

        Args:
            frame: Image BGR
            position: Position du ballon
            trajectory: Trajectoire récente
            confidence: Confiance de la détection

        Returns:
            Image annotée
        """
        result = frame.copy()
        x, y = int(position[0]), int(position[1])

        # Dessiner la traînée du ballon si on a une trajectoire
        if trajectory and len(trajectory) > 1:
            for i in range(1, min(len(trajectory), 15)):
                idx = len(trajectory) - i - 1
                if idx >= 0:
                    pt1 = (int(trajectory[idx][0]), int(trajectory[idx][1]))
                    pt2 = (int(trajectory[idx + 1][0]), int(trajectory[idx + 1][1]))

                    # Opacité décroissante
                    alpha = 0.4 * (1 - i / 15)
                    thickness = max(1, 4 - i // 4)

                    overlay = result.copy()
                    cv2.line(overlay, pt1, pt2, ColorPalette.BALL, thickness, cv2.LINE_AA)
                    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Effet de halo lumineux autour du ballon
        for radius in range(20, 8, -3):
            alpha = 0.15 * (20 - radius) / 12
            overlay = result.copy()
            cv2.circle(overlay, (x, y), radius, ColorPalette.BALL, -1, cv2.LINE_AA)
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

        # Ballon principal - plus gros et visible
        cv2.circle(result, (x, y), 12, ColorPalette.BALL, -1, cv2.LINE_AA)
        cv2.circle(result, (x, y), 12, (0, 0, 0), 2, cv2.LINE_AA)

        # Highlight pour effet 3D
        cv2.circle(result, (x - 3, y - 3), 3, (255, 255, 255), -1, cv2.LINE_AA)

        return result

    def draw_speed_indicator(
        self,
        frame: np.ndarray,
        position: Tuple[float, float],
        speed: float,
        max_speed: float = 35.0
    ) -> np.ndarray:
        """
        Dessiner un indicateur de vitesse

        Args:
            frame: Image BGR
            position: Position pour l'indicateur
            speed: Vitesse actuelle en km/h
            max_speed: Vitesse maximum pour la normalisation

        Returns:
            Image annotée
        """
        result = frame.copy()
        x, y = int(position[0]), int(position[1])

        # Barre de vitesse
        bar_width = 30
        bar_height = 4
        bar_x = x - bar_width // 2
        bar_y = y + 10

        # Fond
        cv2.rectangle(
            result,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (50, 50, 50), -1
        )

        # Barre de progression
        progress = min(speed / max_speed, 1.0)
        progress_width = int(bar_width * progress)

        # Couleur selon la vitesse
        if speed < 7:
            color = (0, 255, 0)  # Vert - marche
        elif speed < 14:
            color = (0, 255, 255)  # Jaune - jogging
        elif speed < 21:
            color = (0, 165, 255)  # Orange - course
        else:
            color = (0, 0, 255)  # Rouge - sprint

        if progress_width > 0:
            cv2.rectangle(
                result,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + bar_height),
                color, -1
            )

        return result

    def draw_minimap(
        self,
        frame: np.ndarray,
        player_positions: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, str],
        ball_position: Tuple[float, float] = None,
        pitch_mapper = None
    ) -> np.ndarray:
        """
        Dessiner une minimap du terrain

        Args:
            frame: Image BGR
            player_positions: Positions des joueurs
            player_teams: Équipes des joueurs
            ball_position: Position du ballon
            pitch_mapper: PitchMapper pour la conversion

        Returns:
            Image avec minimap
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        # Dimensions de la minimap
        map_w = config.visualization.minimap_width
        map_h = config.visualization.minimap_height
        margin = config.visualization.minimap_margin

        # Position selon la configuration
        pos = config.visualization.minimap_position
        if pos == "bottom_right":
            map_x = w - map_w - margin
            map_y = h - map_h - margin
        elif pos == "bottom_left":
            map_x = margin
            map_y = h - map_h - margin
        elif pos == "top_right":
            map_x = w - map_w - margin
            map_y = margin
        else:  # top_left
            map_x = margin
            map_y = margin

        # Créer la minimap
        if self._minimap_cache is None or self._minimap_cache.shape[:2] != (map_h, map_w):
            self._minimap_cache = self._create_minimap_background(map_w, map_h)

        minimap = self._minimap_cache.copy()

        # Ajouter les joueurs
        for player_id, pos_px in player_positions.items():
            # Convertir en coordonnées minimap
            if pitch_mapper and pitch_mapper.homography_matrix is not None:
                pitch_pos = pitch_mapper.transform_point(pos_px)
                if pitch_pos:
                    mx = int(pitch_pos[0] / config.pitch.length * map_w)
                    my = int(pitch_pos[1] / config.pitch.width * map_h)
                else:
                    continue
            else:
                # Estimation simple si pas de calibration
                mx = int(pos_px[0] / w * map_w)
                my = int(pos_px[1] / h * map_h)

            mx = max(0, min(map_w - 1, mx))
            my = max(0, min(map_h - 1, my))

            team = player_teams.get(player_id, "unknown")
            color = get_team_color(team)
            cv2.circle(minimap, (mx, my), 5, color, -1, cv2.LINE_AA)
            cv2.circle(minimap, (mx, my), 5, (255, 255, 255), 1, cv2.LINE_AA)

        # Ajouter le ballon
        if ball_position:
            if pitch_mapper and pitch_mapper.homography_matrix is not None:
                pitch_pos = pitch_mapper.transform_point(ball_position)
                if pitch_pos:
                    bx = int(pitch_pos[0] / config.pitch.length * map_w)
                    by = int(pitch_pos[1] / config.pitch.width * map_h)
                else:
                    bx = int(ball_position[0] / w * map_w)
                    by = int(ball_position[1] / h * map_h)
            else:
                bx = int(ball_position[0] / w * map_w)
                by = int(ball_position[1] / h * map_h)

            bx = max(0, min(map_w - 1, bx))
            by = max(0, min(map_h - 1, by))
            cv2.circle(minimap, (bx, by), 4, ColorPalette.BALL, -1, cv2.LINE_AA)
            cv2.circle(minimap, (bx, by), 4, (0, 0, 0), 1, cv2.LINE_AA)

        # Ajouter le contour de la minimap
        cv2.rectangle(minimap, (0, 0), (map_w - 1, map_h - 1), (255, 255, 255), 1)

        # Placer la minimap sur le frame
        # Avec transparence
        overlay = result.copy()
        overlay[map_y:map_y + map_h, map_x:map_x + map_w] = minimap
        result = cv2.addWeighted(overlay, 0.85, result, 0.15, 0)

        return result

    def _create_minimap_background(self, width: int, height: int) -> np.ndarray:
        """Créer le fond de la minimap"""
        minimap = np.zeros((height, width, 3), dtype=np.uint8)
        minimap[:] = (34, 100, 34)  # Vert foncé

        # Lignes du terrain simplifiées
        line_color = (255, 255, 255)

        # Contour
        cv2.rectangle(minimap, (2, 2), (width - 3, height - 3), line_color, 1)

        # Ligne centrale
        cv2.line(minimap, (width // 2, 2), (width // 2, height - 3), line_color, 1)

        # Cercle central
        cv2.circle(minimap, (width // 2, height // 2), height // 6, line_color, 1)

        # Surfaces de réparation
        pen_w = int(width * 0.16)
        pen_h = int(height * 0.6)
        pen_y = (height - pen_h) // 2
        cv2.rectangle(minimap, (2, pen_y), (pen_w, pen_y + pen_h), line_color, 1)
        cv2.rectangle(minimap, (width - pen_w - 1, pen_y), (width - 3, pen_y + pen_h), line_color, 1)

        return minimap

    def draw_stats_footer(
        self,
        frame: np.ndarray,
        stats: Dict[str, any]
    ) -> np.ndarray:
        """
        Dessiner un footer avec les statistiques du match

        Style broadcast TV professionnel avec possession et passes.

        Args:
            frame: Image BGR
            stats: Dictionnaire avec possession_a, possession_b, passes_a, passes_b

        Returns:
            Image avec footer
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        # Dimensions du footer
        footer_h = 50
        footer_y = h - footer_h

        # Fond semi-transparent
        overlay = result.copy()
        cv2.rectangle(overlay, (0, footer_y), (w, h), (20, 20, 20), -1)
        result = cv2.addWeighted(overlay, 0.85, result, 0.15, 0)

        # Ligne de séparation en haut
        cv2.line(result, (0, footer_y), (w, footer_y), (100, 100, 100), 1)

        # Récupérer les stats
        poss_a = stats.get("possession_a", 50)
        poss_b = stats.get("possession_b", 50)
        passes_a = stats.get("passes_a", 0)
        passes_b = stats.get("passes_b", 0)

        # Couleurs des équipes
        color_a = get_team_color("team_a")
        color_b = get_team_color("team_b")

        # ===== SECTION POSSESSION (centre) =====
        center_x = w // 2
        bar_width = 300
        bar_height = 12
        bar_x = center_x - bar_width // 2
        bar_y = footer_y + 20

        # Titre "POSSESSION"
        cv2.putText(
            result, "POSSESSION",
            (center_x - 45, footer_y + 12),
            self.font, 0.4, (180, 180, 180), 1, cv2.LINE_AA
        )

        # Barre de possession
        # Fond gris
        cv2.rectangle(
            result,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (60, 60, 60), -1
        )

        # Partie équipe A (gauche)
        poss_a_width = int(bar_width * poss_a / 100)
        if poss_a_width > 0:
            cv2.rectangle(
                result,
                (bar_x, bar_y),
                (bar_x + poss_a_width, bar_y + bar_height),
                color_a, -1
            )

        # Partie équipe B (droite)
        poss_b_width = int(bar_width * poss_b / 100)
        if poss_b_width > 0:
            cv2.rectangle(
                result,
                (bar_x + bar_width - poss_b_width, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                color_b, -1
            )

        # Pourcentages
        cv2.putText(
            result, f"{poss_a:.0f}%",
            (bar_x - 45, bar_y + 10),
            self.font, 0.5, color_a, 1, cv2.LINE_AA
        )
        cv2.putText(
            result, f"{poss_b:.0f}%",
            (bar_x + bar_width + 10, bar_y + 10),
            self.font, 0.5, color_b, 1, cv2.LINE_AA
        )

        # ===== SECTION PASSES (gauche et droite) =====
        # Équipe A - Gauche
        section_width = 150
        passes_x_a = 30

        cv2.putText(
            result, "PASSES",
            (passes_x_a, footer_y + 12),
            self.font, 0.35, (150, 150, 150), 1, cv2.LINE_AA
        )

        # Cercle coloré + nombre
        cv2.circle(result, (passes_x_a + 15, footer_y + 30), 8, color_a, -1, cv2.LINE_AA)
        cv2.putText(
            result, str(passes_a),
            (passes_x_a + 30, footer_y + 35),
            self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )

        # Équipe B - Droite
        passes_x_b = w - 80

        cv2.putText(
            result, "PASSES",
            (passes_x_b - 20, footer_y + 12),
            self.font, 0.35, (150, 150, 150), 1, cv2.LINE_AA
        )

        cv2.circle(result, (passes_x_b + 15, footer_y + 30), 8, color_b, -1, cv2.LINE_AA)
        cv2.putText(
            result, str(passes_b),
            (passes_x_b + 30, footer_y + 35),
            self.font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )

        return result

    def draw_stats_panel(
        self,
        frame: np.ndarray,
        stats: Dict[str, any]
    ) -> np.ndarray:
        """
        Dessiner un panneau de statistiques

        Args:
            frame: Image BGR
            stats: Dictionnaire de statistiques

        Returns:
            Image avec panneau
        """
        result = frame.copy()
        h, w = frame.shape[:2]

        # Dimensions du panneau
        panel_w = config.visualization.stats_panel_width
        panel_h = config.visualization.stats_panel_height
        margin = 20

        # Position
        pos = config.visualization.stats_panel_position
        if pos == "top_left":
            px, py = margin, margin
        elif pos == "top_right":
            px, py = w - panel_w - margin, margin
        elif pos == "bottom_left":
            px, py = margin, h - panel_h - margin
        else:
            px, py = w - panel_w - margin, h - panel_h - margin

        # Fond semi-transparent
        overlay = result.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)

        # Bordure
        cv2.rectangle(result, (px, py), (px + panel_w, py + panel_h), (100, 100, 100), 1)

        # Contenu
        y_offset = py + 25
        line_height = 22
        padding = 10

        # Titre
        cv2.putText(
            result, "MATCH STATS",
            (px + padding, y_offset),
            self.font, 0.5, (255, 255, 255), 1
        )
        y_offset += line_height + 5

        # Stats
        stat_items = [
            ("Players", stats.get("player_count", 0)),
            ("Possession", f"{stats.get('possession_a', 50):.0f}% - {stats.get('possession_b', 50):.0f}%"),
            ("Max Speed", f"{stats.get('max_speed', 0):.1f} km/h"),
            ("Total Dist", f"{stats.get('total_distance', 0):.0f}m"),
        ]

        for label, value in stat_items:
            if y_offset + line_height > py + panel_h:
                break

            cv2.putText(
                result, f"{label}:",
                (px + padding, y_offset),
                self.font, 0.4, (180, 180, 180), 1
            )
            cv2.putText(
                result, str(value),
                (px + padding + 80, y_offset),
                self.font, 0.4, (255, 255, 255), 1
            )
            y_offset += line_height

        return result

    def draw_all(
        self,
        frame: np.ndarray,
        tracked_players: List[TrackedObject],
        tracked_ball: Optional[TrackedObject] = None,
        poses: Dict[int, Pose] = None,
        stats: Dict = None,
        pitch_mapper = None,
        draw_trajectories: bool = False,
        draw_minimap: bool = True,
        draw_stats: bool = False,
        draw_skeletons: bool = False,
        draw_formation: bool = True,
        draw_footer: bool = True
    ) -> np.ndarray:
        """
        Dessiner toutes les annotations

        Args:
            frame: Image BGR
            tracked_players: Liste des joueurs trackés
            tracked_ball: Ballon tracké
            poses: Poses par index de joueur
            stats: Statistiques du match (possession_a, possession_b, passes_a, passes_b)
            pitch_mapper: PitchMapper
            draw_trajectories: Dessiner les trajectoires
            draw_minimap: Dessiner la minimap
            draw_stats: Dessiner le panneau de stats
            draw_skeletons: Dessiner les squelettes
            draw_formation: Dessiner les lignes de formation
            draw_footer: Dessiner le footer avec possession et passes

        Returns:
            Image annotée
        """
        result = frame.copy()
        poses = poses or {}

        # Collecter les positions pour la minimap
        player_positions = {}
        player_teams = {}

        # Dessiner les lignes de formation EN PREMIER (sous les cercles)
        if draw_formation:
            result = self.draw_formation_lines(result, tracked_players)

        # Dessiner les joueurs (cercle de couleur sous chaque joueur)
        for player in tracked_players:
            # Cercle de couleur
            result = self.draw_player_circle(
                result,
                player.bbox,
                player.team
            )

            # Squelette (optionnel)
            if draw_skeletons and player.track_id in poses:
                result = self.draw_pose_skeleton(
                    result,
                    poses[player.track_id],
                    player.team
                )

            # Collecter pour minimap
            player_positions[player.track_id] = player.bottom_center
            player_teams[player.track_id] = player.team

        # Minimap (sans le ballon)
        if draw_minimap and player_positions:
            result = self.draw_minimap(
                result,
                player_positions,
                player_teams,
                None,  # Pas de ballon
                pitch_mapper
            )

        # Footer avec stats (possession et passes)
        if draw_footer and stats:
            result = self.draw_stats_footer(result, stats)

        return result
