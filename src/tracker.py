"""
Football Analysis - Multi-Object Tracking avec ByteTrack
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import supervision as sv

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.detector import Detection, DetectionResult


@dataclass
class TrackedObject:
    """Objet tracké avec son historique"""
    track_id: int
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    team: str = "unknown"
    speed: float = 0.0

    # Historique des positions (centre bas pour les joueurs)
    position_history: List[Tuple[float, float]] = field(default_factory=list)

    # Métadonnées
    frames_tracked: int = 0
    last_seen_frame: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        """Centre de la bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Centre bas (position des pieds)"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    def add_position(self, position: Tuple[float, float], max_history: int = 90):
        """Ajouter une position à l'historique"""
        self.position_history.append(position)
        if len(self.position_history) > max_history:
            self.position_history.pop(0)

    def get_trajectory(self, length: int = None) -> List[Tuple[float, float]]:
        """Obtenir la trajectoire récente"""
        if length is None:
            return self.position_history.copy()
        return self.position_history[-length:]


@dataclass
class TrackingResult:
    """Résultat du tracking pour un frame"""
    players: List[TrackedObject] = field(default_factory=list)
    ball: Optional[TrackedObject] = None
    frame_idx: int = 0

    @property
    def player_count(self) -> int:
        return len(self.players)

    @property
    def has_ball(self) -> bool:
        return self.ball is not None

    def get_player_by_id(self, track_id: int) -> Optional[TrackedObject]:
        """Obtenir un joueur par son ID"""
        for player in self.players:
            if player.track_id == track_id:
                return player
        return None


class MultiObjectTracker:
    """
    Tracker multi-objets utilisant ByteTrack via supervision

    Gère le tracking persistant des joueurs et du ballon à travers les frames.
    """

    def __init__(
        self,
        track_thresh: float = None,
        track_buffer: int = None,
        match_thresh: float = None,
        max_history_length: int = None
    ):
        """
        Initialiser le tracker

        Args:
            track_thresh: Seuil de confiance pour le tracking
            track_buffer: Buffer de frames avant suppression
            match_thresh: Seuil de matching
            max_history_length: Longueur max de l'historique
        """
        self.track_thresh = track_thresh or config.tracking.track_thresh
        self.track_buffer = track_buffer or config.tracking.track_buffer
        self.match_thresh = match_thresh or config.tracking.match_thresh
        self.max_history_length = max_history_length or config.tracking.max_history_length

        # Tracker pour les joueurs
        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=30
        )

        # Tracker séparé pour le ballon
        self.ball_tracker = sv.ByteTrack(
            track_activation_threshold=0.1,  # Seuil plus bas pour le ballon
            lost_track_buffer=60,  # Plus de patience pour le ballon
            minimum_matching_threshold=0.5,
            frame_rate=30
        )

        # Historique des objets trackés
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.ball_history: List[Tuple[float, float]] = []

        # Pour l'interpolation du ballon quand non détecté
        self.last_ball_position: Optional[Tuple[float, float]] = None
        self.last_ball_bbox: Optional[Tuple[float, float, float, float]] = None
        self.frames_without_ball: int = 0
        self.max_interpolation_frames: int = 15  # Interpole pendant 0.5s max

        # Compteur de frames
        self.frame_count = 0

    def update(self, detection_result: DetectionResult) -> TrackingResult:
        """
        Mettre à jour le tracking avec de nouvelles détections

        Args:
            detection_result: Résultat de détection

        Returns:
            TrackingResult avec les objets trackés
        """
        self.frame_count += 1

        # Tracker les joueurs
        tracked_players = self._track_players(detection_result.players)

        # Tracker le ballon
        tracked_ball = self._track_ball(detection_result.ball)

        return TrackingResult(
            players=tracked_players,
            ball=tracked_ball,
            frame_idx=self.frame_count
        )

    def _track_players(self, detections: List[Detection]) -> List[TrackedObject]:
        """Tracker les joueurs"""
        if not detections:
            return []

        # Convertir en format supervision
        xyxy = np.array([d.bbox for d in detections])
        confidence = np.array([d.confidence for d in detections])
        class_id = np.array([d.class_id for d in detections])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Appliquer le tracking
        tracked_detections = self.player_tracker.update_with_detections(sv_detections)

        # Convertir en TrackedObject
        tracked_players = []

        if tracked_detections.tracker_id is not None:
            for i in range(len(tracked_detections)):
                track_id = int(tracked_detections.tracker_id[i])
                bbox = tuple(tracked_detections.xyxy[i])
                conf = float(tracked_detections.confidence[i])
                cls_id = int(tracked_detections.class_id[i])

                # Récupérer ou créer l'objet tracké
                if track_id in self.tracked_objects:
                    tracked_obj = self.tracked_objects[track_id]
                    tracked_obj.bbox = bbox
                    tracked_obj.confidence = conf
                    tracked_obj.frames_tracked += 1
                    tracked_obj.last_seen_frame = self.frame_count
                else:
                    tracked_obj = TrackedObject(
                        track_id=track_id,
                        bbox=bbox,
                        confidence=conf,
                        class_id=cls_id,
                        class_name="person",
                        frames_tracked=1,
                        last_seen_frame=self.frame_count
                    )
                    self.tracked_objects[track_id] = tracked_obj

                # Ajouter la position à l'historique
                tracked_obj.add_position(
                    tracked_obj.bottom_center,
                    self.max_history_length
                )

                tracked_players.append(tracked_obj)

        return tracked_players

    def _track_ball(self, detection: Optional[Detection]) -> Optional[TrackedObject]:
        """Tracker le ballon avec interpolation"""
        if detection is None:
            # Pas de détection - essayer d'interpoler
            self.frames_without_ball += 1

            if (self.last_ball_position is not None and
                self.frames_without_ball <= self.max_interpolation_frames and
                len(self.ball_history) >= 2):

                # Interpoler la position basée sur le mouvement récent
                interpolated_pos = self._interpolate_ball_position()

                if interpolated_pos:
                    self.ball_history.append(interpolated_pos)
                    if len(self.ball_history) > self.max_history_length:
                        self.ball_history.pop(0)

                    # Créer une bbox approximative
                    if self.last_ball_bbox:
                        w = self.last_ball_bbox[2] - self.last_ball_bbox[0]
                        h = self.last_ball_bbox[3] - self.last_ball_bbox[1]
                        bbox = (
                            interpolated_pos[0] - w/2,
                            interpolated_pos[1] - h/2,
                            interpolated_pos[0] + w/2,
                            interpolated_pos[1] + h/2
                        )
                    else:
                        bbox = (interpolated_pos[0] - 10, interpolated_pos[1] - 10,
                                interpolated_pos[0] + 10, interpolated_pos[1] + 10)

                    self.last_ball_position = interpolated_pos

                    return TrackedObject(
                        track_id=999,  # ID spécial pour ballon interpolé
                        bbox=bbox,
                        confidence=0.5,  # Confiance réduite pour interpolation
                        class_id=32,
                        class_name="sports ball (interpolé)",
                        position_history=self.ball_history.copy()
                    )

            return None

        # Convertir en format supervision
        xyxy = np.array([detection.bbox])
        confidence = np.array([detection.confidence])
        class_id = np.array([detection.class_id])

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # Appliquer le tracking
        tracked_detections = self.ball_tracker.update_with_detections(sv_detections)

        if tracked_detections.tracker_id is not None and len(tracked_detections) > 0:
            track_id = int(tracked_detections.tracker_id[0])
            bbox = tuple(tracked_detections.xyxy[0])
            conf = float(tracked_detections.confidence[0])

            # Ajouter à l'historique du ballon
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            self.ball_history.append(center)
            if len(self.ball_history) > self.max_history_length:
                self.ball_history.pop(0)

            # Sauvegarder pour interpolation future
            self.last_ball_position = center
            self.last_ball_bbox = bbox
            self.frames_without_ball = 0

            return TrackedObject(
                track_id=track_id,
                bbox=bbox,
                confidence=conf,
                class_id=detection.class_id,
                class_name="sports ball",
                position_history=self.ball_history.copy()
            )

        return None

    def _interpolate_ball_position(self) -> Optional[Tuple[float, float]]:
        """Interpoler la position du ballon basée sur le mouvement récent"""
        if len(self.ball_history) < 2:
            return None

        # Utiliser les dernières positions pour prédire
        p1 = self.ball_history[-2]
        p2 = self.ball_history[-1]

        # Vitesse estimée
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]

        # Réduire progressivement la vitesse (friction)
        decay = 0.9 ** self.frames_without_ball

        # Position prédite
        predicted_x = p2[0] + vx * decay
        predicted_y = p2[1] + vy * decay

        return (predicted_x, predicted_y)

    def get_player_positions(self) -> Dict[int, Tuple[float, float]]:
        """Obtenir les positions actuelles de tous les joueurs"""
        positions = {}
        for track_id, obj in self.tracked_objects.items():
            if obj.last_seen_frame == self.frame_count:
                positions[track_id] = obj.bottom_center
        return positions

    def get_player_trajectories(
        self,
        length: int = None
    ) -> Dict[int, List[Tuple[float, float]]]:
        """Obtenir les trajectoires de tous les joueurs"""
        trajectories = {}
        for track_id, obj in self.tracked_objects.items():
            if obj.position_history:
                trajectories[track_id] = obj.get_trajectory(length)
        return trajectories

    def get_ball_trajectory(self, length: int = None) -> List[Tuple[float, float]]:
        """Obtenir la trajectoire du ballon"""
        if length is None:
            return self.ball_history.copy()
        return self.ball_history[-length:]

    def get_active_track_ids(self) -> List[int]:
        """Obtenir les IDs des tracks actifs"""
        return [
            track_id for track_id, obj in self.tracked_objects.items()
            if obj.last_seen_frame == self.frame_count
        ]

    def reset(self):
        """Réinitialiser le tracker"""
        self.player_tracker = sv.ByteTrack(
            track_activation_threshold=self.track_thresh,
            lost_track_buffer=self.track_buffer,
            minimum_matching_threshold=self.match_thresh,
            frame_rate=30
        )
        self.ball_tracker = sv.ByteTrack(
            track_activation_threshold=0.1,
            lost_track_buffer=60,
            minimum_matching_threshold=0.5,
            frame_rate=30
        )
        self.tracked_objects.clear()
        self.ball_history.clear()
        self.last_ball_position = None
        self.last_ball_bbox = None
        self.frames_without_ball = 0
        self.frame_count = 0

    def get_statistics(self) -> Dict:
        """Obtenir les statistiques du tracking"""
        active_tracks = self.get_active_track_ids()
        return {
            "total_tracks": len(self.tracked_objects),
            "active_tracks": len(active_tracks),
            "frame_count": self.frame_count,
            "ball_positions": len(self.ball_history)
        }


class BallTracker:
    """
    Tracker spécialisé pour le ballon avec interpolation

    Gère les occlusions et les pertes de détection du ballon.
    """

    def __init__(self, max_lost_frames: int = 30):
        """
        Args:
            max_lost_frames: Nombre max de frames sans détection avant reset
        """
        self.max_lost_frames = max_lost_frames
        self.positions: List[Tuple[float, float]] = []
        self.confidences: List[float] = []
        self.frames_since_detection = 0
        self.is_visible = False

    def update(
        self,
        detection: Optional[Detection]
    ) -> Optional[Tuple[float, float]]:
        """
        Mettre à jour avec une nouvelle détection

        Args:
            detection: Détection du ballon ou None

        Returns:
            Position estimée du ballon ou None
        """
        if detection is not None:
            # Nouvelle détection
            position = detection.center
            self.positions.append(position)
            self.confidences.append(detection.confidence)
            self.frames_since_detection = 0
            self.is_visible = True

            # Limiter la taille de l'historique
            if len(self.positions) > 90:
                self.positions.pop(0)
                self.confidences.pop(0)

            return position
        else:
            # Pas de détection
            self.frames_since_detection += 1

            if self.frames_since_detection > self.max_lost_frames:
                self.is_visible = False
                return None

            # Interpoler la position si possible
            if len(self.positions) >= 2:
                return self._interpolate_position()

            return None

    def _interpolate_position(self) -> Optional[Tuple[float, float]]:
        """Interpoler la position du ballon basée sur le mouvement récent"""
        if len(self.positions) < 2:
            return None

        # Utiliser les dernières positions pour prédire
        p1 = self.positions[-2]
        p2 = self.positions[-1]

        # Vitesse estimée
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]

        # Position prédite
        predicted_x = p2[0] + vx * self.frames_since_detection
        predicted_y = p2[1] + vy * self.frames_since_detection

        return (predicted_x, predicted_y)

    def get_trajectory(self, length: int = None) -> List[Tuple[float, float]]:
        """Obtenir la trajectoire"""
        if length is None:
            return self.positions.copy()
        return self.positions[-length:]

    def reset(self):
        """Réinitialiser le tracker"""
        self.positions.clear()
        self.confidences.clear()
        self.frames_since_detection = 0
        self.is_visible = False
