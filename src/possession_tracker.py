"""
Football Analysis - Tracker de possession et passes

Calcule la possession de balle et détecte les passes.
Version améliorée avec meilleure détection des passes.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from collections import deque

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tracker import TrackedObject


@dataclass
class PossessionStats:
    """Statistiques de possession et passes"""
    team_a_frames: int = 0
    team_b_frames: int = 0
    team_a_passes: int = 0
    team_b_passes: int = 0
    total_frames: int = 0

    @property
    def team_a_possession(self) -> float:
        """Pourcentage de possession équipe A"""
        if self.total_frames == 0:
            return 50.0
        return (self.team_a_frames / self.total_frames) * 100

    @property
    def team_b_possession(self) -> float:
        """Pourcentage de possession équipe B"""
        if self.total_frames == 0:
            return 50.0
        return (self.team_b_frames / self.total_frames) * 100


class PossessionTracker:
    """
    Tracker de possession et détection de passes

    Détermine quelle équipe contrôle le ballon et compte les passes.
    Version améliorée avec:
    - Distance de possession plus grande
    - Détection des passes basée sur le mouvement du ballon
    - Cooldown pour éviter les doubles comptages
    """

    def __init__(
        self,
        possession_distance: float = 150.0,  # Augmenté pour mieux détecter
        pass_cooldown_frames: int = 15,  # Attendre 0.5s entre deux passes
        min_pass_distance: float = 80.0  # Distance min du ballon pour une passe
    ):
        """
        Args:
            possession_distance: Distance max pour considérer qu'un joueur a le ballon
            pass_cooldown_frames: Frames minimum entre deux passes détectées
            min_pass_distance: Distance minimum du ballon pour valider une passe
        """
        self.possession_distance = possession_distance
        self.pass_cooldown_frames = pass_cooldown_frames
        self.min_pass_distance = min_pass_distance

        # Stats
        self.stats = PossessionStats()

        # État courant
        self.current_possession: Optional[str] = None
        self.last_possessor_id: Optional[int] = None
        self.last_possessor_team: Optional[str] = None
        self.possession_history: deque = deque(maxlen=10)

        # Pour détection des passes
        self.last_ball_position: Optional[Tuple[float, float]] = None
        self.ball_positions: deque = deque(maxlen=30)
        self.frames_since_last_pass: int = 100  # Commence haut pour permettre la première passe
        self.pass_start_position: Optional[Tuple[float, float]] = None

    def update(
        self,
        players: List[TrackedObject],
        ball: Optional[TrackedObject]
    ) -> Optional[str]:
        """
        Mettre à jour la possession et détecter les passes

        Args:
            players: Liste des joueurs trackés
            ball: Ballon tracké (peut être None)

        Returns:
            Équipe ayant la possession ("team_a", "team_b", ou None)
        """
        self.stats.total_frames += 1
        self.frames_since_last_pass += 1

        if ball is None:
            # Pas de ballon détecté, garder la dernière possession
            if self.current_possession == "team_a":
                self.stats.team_a_frames += 1
            elif self.current_possession == "team_b":
                self.stats.team_b_frames += 1
            return self.current_possession

        ball_pos = ball.center
        self.ball_positions.append(ball_pos)

        # Trouver le joueur le plus proche du ballon (seulement équipes, pas arbitres)
        closest_player = None
        min_distance = float('inf')

        for player in players:
            if player.team not in ["team_a", "team_b"]:
                continue

            # Distance entre le joueur (pieds) et le ballon
            player_pos = player.bottom_center
            distance = np.sqrt(
                (player_pos[0] - ball_pos[0]) ** 2 +
                (player_pos[1] - ball_pos[1]) ** 2
            )

            if distance < min_distance:
                min_distance = distance
                closest_player = player

        # Déterminer la possession
        new_possession = None
        new_possessor_id = None

        if closest_player and min_distance < self.possession_distance:
            new_possession = closest_player.team
            new_possessor_id = closest_player.track_id

            # Détecter une passe
            self._check_for_pass(
                new_possession,
                new_possessor_id,
                ball_pos
            )

            # Mettre à jour le dernier possesseur
            self.last_possessor_id = new_possessor_id
            self.last_possessor_team = new_possession

        # Mettre à jour l'historique de possession
        self.possession_history.append(new_possession)

        # Confirmer le changement de possession (lissage sur 3 frames)
        if len(self.possession_history) >= 3:
            recent = list(self.possession_history)[-3:]
            team_a_count = recent.count("team_a")
            team_b_count = recent.count("team_b")

            if team_a_count >= 2:
                self.current_possession = "team_a"
            elif team_b_count >= 2:
                self.current_possession = "team_b"

        # Compter les frames de possession
        if self.current_possession == "team_a":
            self.stats.team_a_frames += 1
        elif self.current_possession == "team_b":
            self.stats.team_b_frames += 1

        # Sauvegarder la position du ballon
        self.last_ball_position = ball_pos

        return self.current_possession

    def _check_for_pass(
        self,
        new_team: str,
        new_player_id: int,
        ball_pos: Tuple[float, float]
    ):
        """
        Vérifier si une passe a été effectuée

        Une passe est détectée quand:
        1. Le ballon change de joueur au sein de la même équipe
        2. Assez de temps s'est écoulé depuis la dernière passe
        3. Le ballon a parcouru une distance suffisante
        """
        # Vérifier le cooldown
        if self.frames_since_last_pass < self.pass_cooldown_frames:
            return

        # Vérifier qu'on a un possesseur précédent
        if self.last_possessor_id is None:
            self.pass_start_position = ball_pos
            return

        # Vérifier que c'est la même équipe
        if self.last_possessor_team != new_team:
            # Changement d'équipe, pas une passe
            self.pass_start_position = ball_pos
            return

        # Vérifier que c'est un joueur différent
        if self.last_possessor_id == new_player_id:
            return

        # Vérifier la distance parcourue par le ballon
        if self.pass_start_position is not None:
            ball_distance = np.sqrt(
                (ball_pos[0] - self.pass_start_position[0]) ** 2 +
                (ball_pos[1] - self.pass_start_position[1]) ** 2
            )

            if ball_distance >= self.min_pass_distance:
                # PASSE DÉTECTÉE !
                self._register_pass(new_team)
                self.frames_since_last_pass = 0
                self.pass_start_position = ball_pos

    def _register_pass(self, team: str):
        """Enregistrer une passe"""
        if team == "team_a":
            self.stats.team_a_passes += 1
        elif team == "team_b":
            self.stats.team_b_passes += 1

    def get_stats(self) -> Dict:
        """Obtenir les statistiques courantes"""
        return {
            "possession_a": self.stats.team_a_possession,
            "possession_b": self.stats.team_b_possession,
            "passes_a": self.stats.team_a_passes,
            "passes_b": self.stats.team_b_passes,
            "current_possession": self.current_possession
        }

    def reset(self):
        """Réinitialiser le tracker"""
        self.stats = PossessionStats()
        self.current_possession = None
        self.last_possessor_id = None
        self.last_possessor_team = None
        self.possession_history.clear()
        self.last_ball_position = None
        self.ball_positions.clear()
        self.frames_since_last_pass = 100
        self.pass_start_position = None
