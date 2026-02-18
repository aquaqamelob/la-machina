"""
Football Analysis - Détection YOLO des joueurs et du ballon
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from ultralytics import YOLO
import cv2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config, ModelConfig


@dataclass
class Detection:
    """Représente une détection unique"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        """Centre de la bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Centre bas de la bbox (position des pieds)"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, y2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_xyxy(self) -> np.ndarray:
        """Retourner bbox au format xyxy"""
        return np.array(self.bbox)

    def to_xywh(self) -> np.ndarray:
        """Retourner bbox au format xywh (center x, center y, width, height)"""
        x1, y1, x2, y2 = self.bbox
        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1
        ])


@dataclass
class DetectionResult:
    """Résultat complet de détection pour un frame"""
    players: List[Detection] = field(default_factory=list)
    ball: Optional[Detection] = None
    frame_idx: int = 0

    @property
    def player_count(self) -> int:
        return len(self.players)

    @property
    def has_ball(self) -> bool:
        return self.ball is not None


class PlayerBallDetector:
    """
    Détecteur de joueurs et de ballon utilisant YOLO

    Détecte les personnes (joueurs) et le ballon dans les frames vidéo.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence: float = None,
        iou_threshold: float = None,
        device: str = None
    ):
        """
        Initialiser le détecteur

        Args:
            model_path: Chemin vers le modèle YOLO (défaut: config)
            confidence: Seuil de confiance (défaut: config)
            iou_threshold: Seuil IoU pour NMS (défaut: config)
            device: Device pour l'inférence (défaut: config)
        """
        self.model_path = model_path or config.model.detection_model
        self.confidence = confidence or config.model.detection_confidence
        self.ball_confidence = getattr(config.model, 'ball_confidence', 0.2)
        self.iou_threshold = iou_threshold or config.model.iou_threshold
        self.device = device or config.model.device

        # Classes COCO à détecter
        self.person_class_id = config.model.person_class_id
        self.ball_class_id = config.model.ball_class_id

        # Charger le modèle
        self.model = self._load_model()

    def _load_model(self) -> YOLO:
        """Charger le modèle YOLO"""
        try:
            model = YOLO(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Erreur lors du chargement du modèle: {e}")

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Détecter les joueurs et le ballon dans un frame

        Utilise un seuil bas pour détecter le ballon et filtre les joueurs
        avec un seuil plus élevé.

        Args:
            frame: Image BGR

        Returns:
            DetectionResult avec les détections
        """
        # Utiliser le seuil le plus bas pour attraper le ballon
        min_conf = min(self.confidence, self.ball_confidence)

        # Exécuter l'inférence
        results = self.model(
            frame,
            conf=min_conf,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=[self.person_class_id, self.ball_class_id]
        )

        # Parser les résultats
        players = []
        ball = None

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    if cls_id == self.person_class_id:
                        # Appliquer le seuil pour les joueurs
                        if conf >= self.confidence:
                            detection = Detection(
                                bbox=(float(bbox[0]), float(bbox[1]),
                                      float(bbox[2]), float(bbox[3])),
                                confidence=conf,
                                class_id=cls_id,
                                class_name=result.names[cls_id]
                            )
                            players.append(detection)

                    elif cls_id == self.ball_class_id:
                        # Seuil plus bas pour le ballon
                        if conf >= self.ball_confidence:
                            detection = Detection(
                                bbox=(float(bbox[0]), float(bbox[1]),
                                      float(bbox[2]), float(bbox[3])),
                                confidence=conf,
                                class_id=cls_id,
                                class_name=result.names[cls_id]
                            )
                            # Garder la détection du ballon avec la plus haute confiance
                            if ball is None or conf > ball.confidence:
                                ball = detection

        return DetectionResult(players=players, ball=ball)

    def detect_players(self, frame: np.ndarray) -> List[Detection]:
        """
        Détecter uniquement les joueurs

        Args:
            frame: Image BGR

        Returns:
            Liste des détections de joueurs
        """
        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=[self.person_class_id]
        )

        players = []

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())

                    detection = Detection(
                        bbox=(float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        class_id=self.person_class_id,
                        class_name="person"
                    )
                    players.append(detection)

        return players

    def detect_ball(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Détecter uniquement le ballon

        Args:
            frame: Image BGR

        Returns:
            Détection du ballon ou None
        """
        results = self.model(
            frame,
            conf=self.confidence * 0.8,  # Seuil plus bas pour le ballon
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=[self.ball_class_id]
        )

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                # Prendre la détection avec la plus haute confiance
                best_idx = result.boxes.conf.argmax()
                bbox = result.boxes.xyxy[best_idx].cpu().numpy()
                conf = float(result.boxes.conf[best_idx].cpu().numpy())

                return Detection(
                    bbox=(float(bbox[0]), float(bbox[1]),
                          float(bbox[2]), float(bbox[3])),
                    confidence=conf,
                    class_id=self.ball_class_id,
                    class_name="sports ball"
                )

        return None

    def filter_detections_by_area(
        self,
        detections: List[Detection],
        min_area: float = 500,
        max_area: float = 100000
    ) -> List[Detection]:
        """
        Filtrer les détections par aire

        Args:
            detections: Liste des détections
            min_area: Aire minimum
            max_area: Aire maximum

        Returns:
            Détections filtrées
        """
        return [d for d in detections if min_area <= d.area <= max_area]

    def filter_detections_by_aspect_ratio(
        self,
        detections: List[Detection],
        min_ratio: float = 0.3,
        max_ratio: float = 1.5
    ) -> List[Detection]:
        """
        Filtrer les détections par ratio d'aspect (pour les joueurs)

        Args:
            detections: Liste des détections
            min_ratio: Ratio minimum (width/height)
            max_ratio: Ratio maximum

        Returns:
            Détections filtrées
        """
        filtered = []
        for d in detections:
            ratio = d.width / d.height if d.height > 0 else 0
            if min_ratio <= ratio <= max_ratio:
                filtered.append(d)
        return filtered

    def filter_detections_in_region(
        self,
        detections: List[Detection],
        region: Tuple[int, int, int, int]
    ) -> List[Detection]:
        """
        Filtrer les détections dans une région spécifique

        Args:
            detections: Liste des détections
            region: Région (x1, y1, x2, y2)

        Returns:
            Détections dans la région
        """
        x1, y1, x2, y2 = region
        filtered = []
        for d in detections:
            cx, cy = d.center
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                filtered.append(d)
        return filtered

    def get_detections_as_supervision(self, result: DetectionResult):
        """
        Convertir les détections au format supervision

        Args:
            result: DetectionResult

        Returns:
            supervision.Detections object
        """
        import supervision as sv

        if not result.players:
            return sv.Detections.empty()

        xyxy = np.array([d.bbox for d in result.players])
        confidence = np.array([d.confidence for d in result.players])
        class_id = np.array([d.class_id for d in result.players])

        return sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

    def batch_detect(
        self,
        frames: List[np.ndarray]
    ) -> List[DetectionResult]:
        """
        Détection par batch pour de meilleures performances

        Args:
            frames: Liste d'images BGR

        Returns:
            Liste de DetectionResult
        """
        results = self.model(
            frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
            classes=[self.person_class_id, self.ball_class_id]
        )

        detection_results = []

        for result in results:
            players = []
            ball = None

            if result.boxes is not None:
                boxes = result.boxes

                for i in range(len(boxes)):
                    bbox = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    detection = Detection(
                        bbox=(float(bbox[0]), float(bbox[1]),
                              float(bbox[2]), float(bbox[3])),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=result.names[cls_id]
                    )

                    if cls_id == self.person_class_id:
                        players.append(detection)
                    elif cls_id == self.ball_class_id:
                        if ball is None or conf > ball.confidence:
                            ball = detection

            detection_results.append(DetectionResult(players=players, ball=ball))

        return detection_results
