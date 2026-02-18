"""
Football Analysis - Analyse statistique du match
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.tracker import TrackedObject


@dataclass
class PlayerMatchStats:
    """Statistiques d'un joueur pour le match"""
    player_id: int
    team: str = "unknown"

    # Distance et vitesse
    total_distance: float = 0.0  # mètres
    max_speed: float = 0.0  # km/h
    avg_speed: float = 0.0  # km/h
    sprint_count: int = 0

    # Temps d'activité
    standing_time: float = 0.0  # secondes
    walking_time: float = 0.0
    jogging_time: float = 0.0
    running_time: float = 0.0
    sprinting_time: float = 0.0

    # Position
    avg_position: Tuple[float, float] = (0.0, 0.0)
    heatmap_coverage: float = 0.0  # pourcentage

    # Ball interactions (si implémenté)
    ball_touches: int = 0
    passes_attempted: int = 0
    passes_completed: int = 0


@dataclass
class TeamMatchStats:
    """Statistiques d'une équipe"""
    team_name: str

    # Possession
    possession_percentage: float = 0.0
    possession_time: float = 0.0  # secondes

    # Distance
    total_distance: float = 0.0
    avg_distance_per_player: float = 0.0

    # Vitesse
    team_avg_speed: float = 0.0
    team_max_speed: float = 0.0

    # Sprints
    total_sprints: int = 0

    # Formation
    avg_team_width: float = 0.0  # mètres
    avg_team_depth: float = 0.0


@dataclass
class MatchStats:
    """Statistiques globales du match"""
    # Durée
    duration: float = 0.0  # secondes
    frames_analyzed: int = 0

    # Équipes
    team_a_stats: Optional[TeamMatchStats] = None
    team_b_stats: Optional[TeamMatchStats] = None

    # Joueurs
    player_stats: Dict[int, PlayerMatchStats] = field(default_factory=dict)

    # Ball
    ball_distance_traveled: float = 0.0
    ball_possession_changes: int = 0

    # Événements
    events: List[Dict] = field(default_factory=list)


class MatchStatsAnalyzer:
    """
    Analyseur de statistiques de match

    Collecte et calcule les statistiques en temps réel.
    """

    def __init__(self, fps: float = 30.0):
        """
        Initialiser l'analyseur

        Args:
            fps: Images par seconde de la vidéo
        """
        self.fps = fps
        self.dt = 1.0 / fps

        # Stats par joueur
        self.player_stats: Dict[int, PlayerMatchStats] = {}
        self.player_positions_history: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        self.player_speeds_history: Dict[int, List[float]] = defaultdict(list)

        # Stats par équipe
        self.team_a_players: List[int] = []
        self.team_b_players: List[int] = []

        # Possession
        self.possession_history: List[str] = []  # "team_a" ou "team_b"
        self.last_ball_owner: Optional[str] = None

        # Ball tracking
        self.ball_positions: List[Tuple[float, float]] = []

        # Compteurs
        self.frame_count = 0
        self.events: List[Dict] = []

    def update(
        self,
        tracked_players: List[TrackedObject],
        ball_position: Optional[Tuple[float, float]] = None,
        frame_idx: int = 0
    ):
        """
        Mettre à jour les statistiques avec de nouvelles données

        Args:
            tracked_players: Liste des joueurs trackés
            ball_position: Position du ballon
            frame_idx: Index du frame
        """
        self.frame_count = frame_idx + 1

        # Mettre à jour les stats de chaque joueur
        for player in tracked_players:
            self._update_player_stats(player)

        # Mettre à jour la possession
        if ball_position:
            self._update_possession(tracked_players, ball_position)
            self.ball_positions.append(ball_position)

    def _update_player_stats(self, player: TrackedObject):
        """Mettre à jour les stats d'un joueur"""
        player_id = player.track_id

        # Initialiser si nouveau joueur
        if player_id not in self.player_stats:
            self.player_stats[player_id] = PlayerMatchStats(
                player_id=player_id,
                team=player.team
            )

            # Ajouter aux listes d'équipe
            if player.team == "team_a" and player_id not in self.team_a_players:
                self.team_a_players.append(player_id)
            elif player.team == "team_b" and player_id not in self.team_b_players:
                self.team_b_players.append(player_id)

        stats = self.player_stats[player_id]
        stats.team = player.team

        # Mettre à jour vitesse
        speed = player.speed
        self.player_speeds_history[player_id].append(speed)

        if speed > stats.max_speed:
            stats.max_speed = speed

        # Classifier l'activité
        if speed < 2.0:
            stats.standing_time += self.dt
        elif speed < 7.0:
            stats.walking_time += self.dt
        elif speed < 14.0:
            stats.jogging_time += self.dt
        elif speed < 21.0:
            stats.running_time += self.dt
        else:
            stats.sprinting_time += self.dt
            # Compter les sprints
            speeds = self.player_speeds_history[player_id]
            if len(speeds) >= 2 and speeds[-2] < 21.0:
                stats.sprint_count += 1

        # Ajouter position à l'historique
        self.player_positions_history[player_id].append(player.bottom_center)

    def _update_possession(
        self,
        tracked_players: List[TrackedObject],
        ball_position: Tuple[float, float]
    ):
        """Déterminer et mettre à jour la possession"""
        # Trouver le joueur le plus proche du ballon
        min_dist = float('inf')
        closest_player = None

        for player in tracked_players:
            dist = np.sqrt(
                (player.bottom_center[0] - ball_position[0])**2 +
                (player.bottom_center[1] - ball_position[1])**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_player = player

        # Si assez proche, attribuer la possession
        possession_threshold = 50  # pixels
        if closest_player and min_dist < possession_threshold:
            current_owner = closest_player.team
            self.possession_history.append(current_owner)

            # Détecter changement de possession
            if self.last_ball_owner and current_owner != self.last_ball_owner:
                if current_owner in ["team_a", "team_b"]:
                    self.events.append({
                        "type": "possession_change",
                        "frame": self.frame_count,
                        "from": self.last_ball_owner,
                        "to": current_owner
                    })

            self.last_ball_owner = current_owner
        else:
            self.possession_history.append("contested")

    def calculate_possession_percentage(self) -> Dict[str, float]:
        """Calculer les pourcentages de possession"""
        if not self.possession_history:
            return {"team_a": 50.0, "team_b": 50.0}

        team_a_count = self.possession_history.count("team_a")
        team_b_count = self.possession_history.count("team_b")
        total = team_a_count + team_b_count

        if total == 0:
            return {"team_a": 50.0, "team_b": 50.0}

        return {
            "team_a": (team_a_count / total) * 100,
            "team_b": (team_b_count / total) * 100
        }

    def calculate_average_positions(self) -> Dict[int, Tuple[float, float]]:
        """Calculer les positions moyennes des joueurs"""
        avg_positions = {}
        for player_id, positions in self.player_positions_history.items():
            if positions:
                x_avg = np.mean([p[0] for p in positions])
                y_avg = np.mean([p[1] for p in positions])
                avg_positions[player_id] = (x_avg, y_avg)
        return avg_positions

    def calculate_distances(self) -> Dict[int, float]:
        """Calculer les distances totales parcourues"""
        distances = {}
        for player_id, positions in self.player_positions_history.items():
            if len(positions) >= 2:
                total_dist = 0.0
                for i in range(1, len(positions)):
                    dx = positions[i][0] - positions[i-1][0]
                    dy = positions[i][1] - positions[i-1][1]
                    total_dist += np.sqrt(dx**2 + dy**2)
                # Convertir en mètres (approximation)
                distances[player_id] = total_dist / config.speed.pixels_per_meter
            else:
                distances[player_id] = 0.0
        return distances

    def calculate_team_formation(
        self,
        team: str
    ) -> Dict[str, float]:
        """Calculer les métriques de formation d'une équipe"""
        player_ids = self.team_a_players if team == "team_a" else self.team_b_players

        if not player_ids:
            return {"width": 0.0, "depth": 0.0}

        # Positions moyennes des joueurs de l'équipe
        positions = []
        for pid in player_ids:
            if pid in self.player_positions_history:
                hist = self.player_positions_history[pid]
                if hist:
                    positions.append((np.mean([p[0] for p in hist[-30:]]),
                                     np.mean([p[1] for p in hist[-30:]])))

        if len(positions) < 2:
            return {"width": 0.0, "depth": 0.0}

        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        width = (max(x_coords) - min(x_coords)) / config.speed.pixels_per_meter
        depth = (max(y_coords) - min(y_coords)) / config.speed.pixels_per_meter

        return {"width": width, "depth": depth}

    def get_fastest_player(self) -> Optional[Tuple[int, float]]:
        """Obtenir le joueur ayant atteint la vitesse max"""
        max_speed = 0.0
        fastest_id = None

        for player_id, stats in self.player_stats.items():
            if stats.max_speed > max_speed:
                max_speed = stats.max_speed
                fastest_id = player_id

        return (fastest_id, max_speed) if fastest_id else None

    def get_most_active_player(self) -> Optional[Tuple[int, float]]:
        """Obtenir le joueur ayant le plus de distance"""
        distances = self.calculate_distances()
        if not distances:
            return None

        max_id = max(distances, key=distances.get)
        return (max_id, distances[max_id])

    def get_sprint_leaders(self, top_n: int = 5) -> List[Tuple[int, int]]:
        """Obtenir les joueurs avec le plus de sprints"""
        sprints = [(pid, stats.sprint_count) for pid, stats in self.player_stats.items()]
        sprints.sort(key=lambda x: x[1], reverse=True)
        return sprints[:top_n]

    def generate_report(self) -> MatchStats:
        """Générer le rapport complet du match"""
        # Calculer les stats finales
        possession = self.calculate_possession_percentage()
        distances = self.calculate_distances()
        avg_positions = self.calculate_average_positions()

        # Stats par joueur
        for player_id, stats in self.player_stats.items():
            stats.total_distance = distances.get(player_id, 0.0)
            stats.avg_position = avg_positions.get(player_id, (0.0, 0.0))

            speeds = self.player_speeds_history.get(player_id, [])
            if speeds:
                stats.avg_speed = np.mean(speeds)

        # Stats équipe A
        team_a_stats = TeamMatchStats(team_name="team_a")
        team_a_stats.possession_percentage = possession["team_a"]
        team_a_stats.total_distance = sum(
            distances.get(pid, 0.0) for pid in self.team_a_players
        )
        if self.team_a_players:
            team_a_stats.avg_distance_per_player = (
                team_a_stats.total_distance / len(self.team_a_players)
            )
            team_a_stats.total_sprints = sum(
                self.player_stats[pid].sprint_count
                for pid in self.team_a_players
                if pid in self.player_stats
            )
            team_a_max_speeds = [
                self.player_stats[pid].max_speed
                for pid in self.team_a_players
                if pid in self.player_stats
            ]
            if team_a_max_speeds:
                team_a_stats.team_max_speed = max(team_a_max_speeds)

        formation_a = self.calculate_team_formation("team_a")
        team_a_stats.avg_team_width = formation_a["width"]
        team_a_stats.avg_team_depth = formation_a["depth"]

        # Stats équipe B
        team_b_stats = TeamMatchStats(team_name="team_b")
        team_b_stats.possession_percentage = possession["team_b"]
        team_b_stats.total_distance = sum(
            distances.get(pid, 0.0) for pid in self.team_b_players
        )
        if self.team_b_players:
            team_b_stats.avg_distance_per_player = (
                team_b_stats.total_distance / len(self.team_b_players)
            )
            team_b_stats.total_sprints = sum(
                self.player_stats[pid].sprint_count
                for pid in self.team_b_players
                if pid in self.player_stats
            )
            team_b_max_speeds = [
                self.player_stats[pid].max_speed
                for pid in self.team_b_players
                if pid in self.player_stats
            ]
            if team_b_max_speeds:
                team_b_stats.team_max_speed = max(team_b_max_speeds)

        formation_b = self.calculate_team_formation("team_b")
        team_b_stats.avg_team_width = formation_b["width"]
        team_b_stats.avg_team_depth = formation_b["depth"]

        # Ball distance
        ball_distance = 0.0
        if len(self.ball_positions) >= 2:
            for i in range(1, len(self.ball_positions)):
                dx = self.ball_positions[i][0] - self.ball_positions[i-1][0]
                dy = self.ball_positions[i][1] - self.ball_positions[i-1][1]
                ball_distance += np.sqrt(dx**2 + dy**2)
            ball_distance /= config.speed.pixels_per_meter

        # Assembler le rapport
        return MatchStats(
            duration=self.frame_count / self.fps,
            frames_analyzed=self.frame_count,
            team_a_stats=team_a_stats,
            team_b_stats=team_b_stats,
            player_stats=self.player_stats,
            ball_distance_traveled=ball_distance,
            ball_possession_changes=len([e for e in self.events if e["type"] == "possession_change"]),
            events=self.events
        )

    def get_live_stats(self) -> Dict:
        """Obtenir les statistiques en direct pour l'affichage"""
        possession = self.calculate_possession_percentage()

        # Vitesse max actuelle
        current_speeds = []
        for pid, stats in self.player_stats.items():
            speeds = self.player_speeds_history.get(pid, [])
            if speeds:
                current_speeds.append(speeds[-1])

        max_current_speed = max(current_speeds) if current_speeds else 0.0

        # Distance totale
        distances = self.calculate_distances()
        total_distance = sum(distances.values())

        return {
            "player_count": len(self.player_stats),
            "possession_a": possession["team_a"],
            "possession_b": possession["team_b"],
            "max_speed": max_current_speed,
            "total_distance": total_distance,
            "frame": self.frame_count,
            "duration": self.frame_count / self.fps
        }

    def export_to_json(self, output_path: str):
        """Exporter les statistiques en JSON"""
        report = self.generate_report()

        # Convertir en dict serializable
        data = {
            "duration": report.duration,
            "frames_analyzed": report.frames_analyzed,
            "ball_distance": report.ball_distance_traveled,
            "possession_changes": report.ball_possession_changes,
            "team_a": {
                "possession": report.team_a_stats.possession_percentage if report.team_a_stats else 0,
                "total_distance": report.team_a_stats.total_distance if report.team_a_stats else 0,
                "sprints": report.team_a_stats.total_sprints if report.team_a_stats else 0,
                "max_speed": report.team_a_stats.team_max_speed if report.team_a_stats else 0,
            },
            "team_b": {
                "possession": report.team_b_stats.possession_percentage if report.team_b_stats else 0,
                "total_distance": report.team_b_stats.total_distance if report.team_b_stats else 0,
                "sprints": report.team_b_stats.total_sprints if report.team_b_stats else 0,
                "max_speed": report.team_b_stats.team_max_speed if report.team_b_stats else 0,
            },
            "players": {
                str(pid): {
                    "team": stats.team,
                    "distance": stats.total_distance,
                    "max_speed": stats.max_speed,
                    "avg_speed": stats.avg_speed,
                    "sprints": stats.sprint_count
                }
                for pid, stats in report.player_stats.items()
            },
            "events": report.events
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def reset(self):
        """Réinitialiser l'analyseur"""
        self.player_stats.clear()
        self.player_positions_history.clear()
        self.player_speeds_history.clear()
        self.team_a_players.clear()
        self.team_b_players.clear()
        self.possession_history.clear()
        self.ball_positions.clear()
        self.events.clear()
        self.frame_count = 0
        self.last_ball_owner = None
