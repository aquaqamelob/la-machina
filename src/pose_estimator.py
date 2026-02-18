"""
Football Analysis - Estimation de pose avec YOLO
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from ultralytics import YOLO

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, COCO_KEYPOINTS, SKELETON_CONNECTIONS
from src.detector import Detection


@dataclass
class Keypoint:
    """Un point clé du squelette"""
    name: str
    x: float
    y: float
    confidence: float

    @property
    def point(self) -> Tuple[float, float]:
        return (self.x, self.y)

    @property
    def is_visible(self) -> bool:
        return self.confidence > 0.5


@dataclass
class Pose:
    """Pose complète d'une personne"""
    keypoints: Dict[str, Keypoint] = field(default_factory=dict)
    bbox: Optional[Tuple[float, float, float, float]] = None
    confidence: float = 0.0

    def get_keypoint(self, name: str) -> Optional[Keypoint]:
        """Obtenir un keypoint par son nom"""
        return self.keypoints.get(name)

    def get_point(self, name: str) -> Optional[Tuple[float, float]]:
        """Obtenir les coordonnées d'un keypoint"""
        kp = self.keypoints.get(name)
        if kp and kp.is_visible:
            return kp.point
        return None

    @property
    def is_valid(self) -> bool:
        """Vérifier si la pose est valide (suffisamment de keypoints)"""
        visible_count = sum(1 for kp in self.keypoints.values() if kp.is_visible)
        return visible_count >= 5

    def get_all_visible_points(self) -> List[Tuple[str, Tuple[float, float]]]:
        """Obtenir tous les points visibles"""
        return [
            (name, kp.point)
            for name, kp in self.keypoints.items()
            if kp.is_visible
        ]


class PoseEstimator:
    """
    Estimateur de pose utilisant YOLO-pose

    Détecte les keypoints du corps des joueurs.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = None,
        device: str = None
    ):
        """
        Initialiser l'estimateur de pose

        Args:
            model_path: Chemin vers le modèle YOLO-pose
            confidence: Seuil de confiance
            device: Device pour l'inférence
        """
        self.model_path = model_path or config.model.pose_model
        self.confidence = confidence or config.model.pose_confidence
        self.device = device or config.model.device

        # Charger le modèle
        self.model = self._load_model()

        # Mapping des keypoints COCO
        self.keypoint_names = list(COCO_KEYPOINTS.keys())

    def _load_model(self) -> YOLO:
        """Charger le modèle YOLO-pose"""
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle pose: {e}")

    def estimate_poses(self, frame: np.ndarray) -> List[Pose]:
        """
        Estimer les poses de toutes les personnes dans un frame

        Args:
            frame: Image BGR

        Returns:
            Liste des poses détectées
        """
        results = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        poses = []

        if results and len(results) > 0:
            result = results[0]

            if result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes

                for i in range(len(keypoints_data)):
                    pose = self._parse_keypoints(keypoints_data[i])

                    # Ajouter la bbox si disponible
                    if boxes is not None and i < len(boxes):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        pose.bbox = (
                            float(bbox[0]), float(bbox[1]),
                            float(bbox[2]), float(bbox[3])
                        )
                        pose.confidence = float(boxes.conf[i].cpu().numpy())

                    if pose.is_valid:
                        poses.append(pose)

        return poses

    def estimate_pose_for_detection(
        self,
        frame: np.ndarray,
        detection: Detection
    ) -> Optional[Pose]:
        """
        Estimer la pose pour une détection spécifique

        Args:
            frame: Image BGR
            detection: Détection d'un joueur

        Returns:
            Pose ou None
        """
        # Extraire le ROI
        x1, y1, x2, y2 = map(int, detection.bbox)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        # Estimer la pose sur le ROI
        results = self.model(
            roi,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        if results and len(results) > 0:
            result = results[0]
            if result.keypoints is not None and len(result.keypoints.data) > 0:
                keypoints_data = result.keypoints.data.cpu().numpy()[0]
                pose = self._parse_keypoints(keypoints_data)

                # Ajuster les coordonnées au frame original
                for name, kp in pose.keypoints.items():
                    kp.x += x1
                    kp.y += y1

                pose.bbox = detection.bbox
                pose.confidence = detection.confidence

                return pose

        return None

    def _parse_keypoints(self, keypoints_data: np.ndarray) -> Pose:
        """Parser les données de keypoints"""
        pose = Pose()

        for i, name in enumerate(self.keypoint_names):
            if i < len(keypoints_data):
                x, y, conf = keypoints_data[i]
                pose.keypoints[name] = Keypoint(
                    name=name,
                    x=float(x),
                    y=float(y),
                    confidence=float(conf)
                )

        return pose

    def get_player_orientation(self, pose: Pose) -> Optional[float]:
        """
        Calculer l'orientation d'un joueur en degrés

        Args:
            pose: Pose du joueur

        Returns:
            Angle en degrés (0-360) ou None
        """
        # Utiliser les épaules et hanches pour déterminer l'orientation
        left_shoulder = pose.get_point('left_shoulder')
        right_shoulder = pose.get_point('right_shoulder')
        left_hip = pose.get_point('left_hip')
        right_hip = pose.get_point('right_hip')

        if left_shoulder and right_shoulder:
            # Direction basée sur les épaules
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]

            # Perpendiculaire = direction de face
            angle = np.degrees(np.arctan2(-dx, dy))  # Perpendiculaire
            return angle % 360

        if left_hip and right_hip:
            # Alternative avec les hanches
            dx = right_hip[0] - left_hip[0]
            dy = right_hip[1] - left_hip[1]
            angle = np.degrees(np.arctan2(-dx, dy))
            return angle % 360

        return None

    def is_player_running(
        self,
        pose_history: List[Pose],
        threshold: float = 50.0
    ) -> bool:
        """
        Déterminer si un joueur court basé sur son historique de poses

        Args:
            pose_history: Historique des poses récentes
            threshold: Seuil de mouvement des genoux

        Returns:
            True si le joueur semble courir
        """
        if len(pose_history) < 3:
            return False

        # Analyser le mouvement des genoux
        knee_movements = []

        for i in range(1, len(pose_history)):
            prev_pose = pose_history[i - 1]
            curr_pose = pose_history[i]

            prev_left = prev_pose.get_point('left_knee')
            curr_left = curr_pose.get_point('left_knee')
            prev_right = prev_pose.get_point('right_knee')
            curr_right = curr_pose.get_point('right_knee')

            movement = 0.0
            if prev_left and curr_left:
                movement += abs(curr_left[1] - prev_left[1])
            if prev_right and curr_right:
                movement += abs(curr_right[1] - prev_right[1])

            knee_movements.append(movement)

        avg_movement = np.mean(knee_movements) if knee_movements else 0
        return avg_movement > threshold

    def get_body_center(self, pose: Pose) -> Optional[Tuple[float, float]]:
        """Obtenir le centre du corps"""
        points = []

        for name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            pt = pose.get_point(name)
            if pt:
                points.append(pt)

        if len(points) >= 2:
            x = np.mean([p[0] for p in points])
            y = np.mean([p[1] for p in points])
            return (x, y)

        return None

    def get_skeleton_lines(
        self,
        pose: Pose
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Obtenir les lignes du squelette pour le dessin

        Args:
            pose: Pose du joueur

        Returns:
            Liste de paires de points (start, end)
        """
        lines = []

        for start_name, end_name in SKELETON_CONNECTIONS:
            start_point = pose.get_point(start_name)
            end_point = pose.get_point(end_name)

            if start_point and end_point:
                lines.append((start_point, end_point))

        return lines

    def batch_estimate(self, frames: List[np.ndarray]) -> List[List[Pose]]:
        """
        Estimation de pose en batch

        Args:
            frames: Liste d'images BGR

        Returns:
            Liste de listes de poses
        """
        results = self.model(
            frames,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        all_poses = []

        for result in results:
            poses = []
            if result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
                boxes = result.boxes

                for i in range(len(keypoints_data)):
                    pose = self._parse_keypoints(keypoints_data[i])

                    if boxes is not None and i < len(boxes):
                        bbox = boxes.xyxy[i].cpu().numpy()
                        pose.bbox = (
                            float(bbox[0]), float(bbox[1]),
                            float(bbox[2]), float(bbox[3])
                        )
                        pose.confidence = float(boxes.conf[i].cpu().numpy())

                    if pose.is_valid:
                        poses.append(pose)

            all_poses.append(poses)

        return all_poses


def match_poses_to_detections(
    poses: List[Pose],
    detections: List[Detection],
    iou_threshold: float = 0.3
) -> Dict[int, Pose]:
    """
    Associer les poses aux détections par IoU

    Args:
        poses: Liste des poses
        detections: Liste des détections
        iou_threshold: Seuil d'IoU minimum

    Returns:
        Dictionnaire {index_detection: pose}
    """
    from utils.geometry import bbox_iou

    matches = {}

    for det_idx, detection in enumerate(detections):
        best_iou = 0.0
        best_pose = None

        for pose in poses:
            if pose.bbox is not None:
                iou = bbox_iou(detection.bbox, pose.bbox)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pose = pose

        if best_pose is not None:
            matches[det_idx] = best_pose

    return matches
