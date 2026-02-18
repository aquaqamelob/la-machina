"""
Football Analysis - Calcul des vitesses et distances
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils.geometry import calculate_distance, moving_average


@dataclass
class PlayerStats:
    """Statistiques d'un joueur"""
    player_id: int
    total_distance: float = 0.0  # mètres
    max_speed: float = 0.0  # km/h
    avg_speed: float = 0.0  # km/h
    current_speed: float = 0.0  # km/h
    sprint_count: int = 0
    walking_time: float = 0.0  # secondes
    jogging_time: float = 0.0
    running_time: float = 0.0
    sprinting_time: float = 0.0

    # Historique des vitesses
    speed_history: List[float] = field(default_factory=list)


class SpeedDistanceCalculator:
    """
    Calculateur de vitesse et distance pour les joueurs

    Convertit les positions en pixels vers des mesures réelles (mètres, km/h).
    """

    def __init__(
        self,
        fps: float = 30.0,
        pixels_per_meter: float = None,
        smoothing_window: int = None
    ):
        """
        Initialiser le calculateur

        Args:
            fps: Images par seconde de la vidéo
            pixels_per_meter: Ratio pixels/mètre pour la conversion
            smoothing_window: Taille de la fenêtre de lissage
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter or config.speed.pixels_per_meter
        self.smoothing_window = smoothing_window or config.speed.smoothing_window

        # Seuils de vitesse
        self.walking_threshold = config.speed.walking_threshold
        self.jogging_threshold = config.speed.jogging_threshold
        self.running_threshold = config.speed.running_threshold
        self.sprint_threshold = config.speed.sprint_threshold
        self.max_realistic_speed = config.speed.max_realistic_speed

        # Historique par joueur
        self.position_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.speed_history: Dict[int, List[float]] = defaultdict(list)
        self.player_stats: Dict[int, PlayerStats] = {}

        # Temps entre frames
        self.dt = 1.0 / fps

    def update(
        self,
        player_id: int,
        position: Tuple[float, float]
    ) -> float:
        """
        Mettre à jour avec une nouvelle position et calculer la vitesse

        Args:
            player_id: ID du joueur
            position: Position actuelle (x, y) en pixels

        Returns:
            Vitesse instantanée en km/h
        """
        history = self.position_history[player_id]
        history.append(position)

        # Limiter la taille de l'historique
        max_history = self.smoothing_window * 2
        if len(history) > max_history:
            history.pop(0)

        # Calculer la vitesse
        speed = self._calculate_instant_speed(player_id)

        # Mettre à jour les statistiques
        self._update_stats(player_id, speed)

        return speed

    def _calculate_instant_speed(self, player_id: int) -> float:
        """Calculer la vitesse instantanée"""
        history = self.position_history[player_id]

        if len(history) < 2:
            return 0.0

        # Utiliser la moyenne mobile pour lisser
        if len(history) >= self.smoothing_window:
            # Calculer la distance sur la fenêtre de lissage
            start_pos = history[-self.smoothing_window]
            end_pos = history[-1]
            time_elapsed = self.dt * (self.smoothing_window - 1)
        else:
            start_pos = history[0]
            end_pos = history[-1]
            time_elapsed = self.dt * (len(history) - 1)

        if time_elapsed <= 0:
            return 0.0

        # Distance en pixels
        distance_pixels = calculate_distance(start_pos, end_pos)

        # Convertir en mètres
        distance_meters = distance_pixels / self.pixels_per_meter

        # Vitesse en m/s puis km/h
        speed_ms = distance_meters / time_elapsed
        speed_kmh = speed_ms * 3.6

        # Limiter aux valeurs réalistes
        speed_kmh = min(speed_kmh, self.max_realistic_speed)
        speed_kmh = max(0.0, speed_kmh)

        # Stocker dans l'historique
        self.speed_history[player_id].append(speed_kmh)
        if len(self.speed_history[player_id]) > 90:
            self.speed_history[player_id].pop(0)

        return speed_kmh

    def _update_stats(self, player_id: int, speed: float):
        """Mettre à jour les statistiques du joueur"""
        if player_id not in self.player_stats:
            self.player_stats[player_id] = PlayerStats(player_id=player_id)

        stats = self.player_stats[player_id]
        stats.current_speed = speed
        stats.speed_history.append(speed)

        # Mise à jour de la vitesse max
        if speed > stats.max_speed:
            stats.max_speed = speed

        # Mise à jour de la distance
        history = self.position_history[player_id]
        if len(history) >= 2:
            distance_pixels = calculate_distance(history[-2], history[-1])
            distance_meters = distance_pixels / self.pixels_per_meter
            stats.total_distance += distance_meters

        # Classification de l'activité
        if speed < self.walking_threshold:
            stats.walking_time += self.dt
        elif speed < self.jogging_threshold:
            stats.jogging_time += self.dt
        elif speed < self.running_threshold:
            stats.running_time += self.dt
        else:
            stats.sprinting_time += self.dt
            # Compter les sprints (début de sprint)
            if len(stats.speed_history) >= 2:
                if stats.speed_history[-2] < self.sprint_threshold:
                    stats.sprint_count += 1

        # Moyenne des vitesses
        if stats.speed_history:
            stats.avg_speed = np.mean(stats.speed_history)

    def get_speed(self, player_id: int) -> float:
        """Obtenir la vitesse actuelle d'un joueur"""
        if player_id in self.player_stats:
            return self.player_stats[player_id].current_speed
        return 0.0

    def get_distance(self, player_id: int) -> float:
        """Obtenir la distance totale parcourue par un joueur"""
        if player_id in self.player_stats:
            return self.player_stats[player_id].total_distance
        return 0.0

    def get_player_stats(self, player_id: int) -> Optional[PlayerStats]:
        """Obtenir toutes les statistiques d'un joueur"""
        return self.player_stats.get(player_id)

    def get_all_stats(self) -> Dict[int, PlayerStats]:
        """Obtenir les statistiques de tous les joueurs"""
        return self.player_stats.copy()

    def get_smoothed_speed(self, player_id: int, window: int = 5) -> float:
        """Obtenir la vitesse lissée d'un joueur"""
        history = self.speed_history.get(player_id, [])
        if not history:
            return 0.0
        return np.mean(history[-window:])

    def get_speed_classification(self, speed: float) -> str:
        """
        Classifier une vitesse

        Args:
            speed: Vitesse en km/h

        Returns:
            "standing", "walking", "jogging", "running", ou "sprinting"
        """
        if speed < 2.0:
            return "standing"
        elif speed < self.walking_threshold:
            return "walking"
        elif speed < self.jogging_threshold:
            return "jogging"
        elif speed < self.running_threshold:
            return "running"
        else:
            return "sprinting"

    def calibrate_scale(
        self,
        pixel_distance: float,
        real_distance: float
    ):
        """
        Calibrer l'échelle pixels/mètres

        Args:
            pixel_distance: Distance en pixels
            real_distance: Distance réelle en mètres
        """
        if real_distance > 0:
            self.pixels_per_meter = pixel_distance / real_distance

    def calibrate_from_pitch_lines(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        known_distance: float
    ):
        """
        Calibrer l'échelle à partir de lignes de terrain connues

        Args:
            point1: Premier point en pixels
            point2: Deuxième point en pixels
            known_distance: Distance réelle connue en mètres
        """
        pixel_distance = calculate_distance(point1, point2)
        self.calibrate_scale(pixel_distance, known_distance)

    def reset_player(self, player_id: int):
        """Réinitialiser les données d'un joueur"""
        if player_id in self.position_history:
            del self.position_history[player_id]
        if player_id in self.speed_history:
            del self.speed_history[player_id]
        if player_id in self.player_stats:
            del self.player_stats[player_id]

    def reset_all(self):
        """Réinitialiser toutes les données"""
        self.position_history.clear()
        self.speed_history.clear()
        self.player_stats.clear()

    def get_team_distance(self, team_player_ids: List[int]) -> float:
        """Obtenir la distance totale d'une équipe"""
        total = 0.0
        for player_id in team_player_ids:
            total += self.get_distance(player_id)
        return total

    def get_team_avg_speed(self, team_player_ids: List[int]) -> float:
        """Obtenir la vitesse moyenne d'une équipe"""
        speeds = []
        for player_id in team_player_ids:
            if player_id in self.player_stats:
                speeds.append(self.player_stats[player_id].current_speed)
        return np.mean(speeds) if speeds else 0.0

    def get_fastest_player(self) -> Optional[Tuple[int, float]]:
        """Obtenir le joueur le plus rapide actuellement"""
        if not self.player_stats:
            return None

        max_speed = 0.0
        fastest_id = None

        for player_id, stats in self.player_stats.items():
            if stats.current_speed > max_speed:
                max_speed = stats.current_speed
                fastest_id = player_id

        if fastest_id is not None:
            return (fastest_id, max_speed)
        return None

    def get_summary_stats(self) -> Dict:
        """Obtenir un résumé des statistiques"""
        if not self.player_stats:
            return {}

        all_speeds = [s.current_speed for s in self.player_stats.values()]
        all_distances = [s.total_distance for s in self.player_stats.values()]
        all_max_speeds = [s.max_speed for s in self.player_stats.values()]

        return {
            "player_count": len(self.player_stats),
            "avg_current_speed": np.mean(all_speeds) if all_speeds else 0.0,
            "max_current_speed": max(all_speeds) if all_speeds else 0.0,
            "total_distance": sum(all_distances),
            "avg_distance_per_player": np.mean(all_distances) if all_distances else 0.0,
            "overall_max_speed": max(all_max_speeds) if all_max_speeds else 0.0,
        }
