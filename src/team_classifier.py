"""
Football Analysis - Classification des équipes par couleur de maillot
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from collections import Counter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from src.detector import Detection


@dataclass
class TeamColors:
    """Couleurs représentatives d'une équipe"""
    primary: Tuple[int, int, int]  # Couleur dominante BGR
    secondary: Optional[Tuple[int, int, int]] = None
    name: str = "unknown"


class TeamClassifier:
    """
    Classificateur d'équipes basé sur la couleur des maillots

    Utilise K-means clustering pour extraire les couleurs dominantes
    et classifier les joueurs en équipes.
    Détecte explicitement les arbitres par leur couleur (orange, jaune fluo).
    """

    # Plages de couleurs typiques des arbitres en HSV
    # Orange: H=10-25, S>100, V>100
    # Jaune fluo: H=25-35, S>100, V>150
    # Rose fluo: H=160-180, S>50, V>150
    REFEREE_HSV_RANGES = [
        # (H_min, H_max, S_min, V_min) - Orange
        (5, 25, 100, 100),
        # Jaune / Jaune-vert fluo
        (25, 45, 80, 150),
        # Rose / Magenta fluo
        (150, 180, 50, 150),
    ]

    def __init__(
        self,
        n_clusters: int = 3,
        sample_ratio: float = 0.5,
        exclude_ratio: float = 0.2
    ):
        """
        Initialiser le classificateur

        Args:
            n_clusters: Nombre de clusters pour K-means
            sample_ratio: Ratio de pixels à échantillonner
            exclude_ratio: Ratio de la bbox à exclure (haut et bas pour éviter tête/pieds)
        """
        self.n_clusters = n_clusters
        self.sample_ratio = sample_ratio
        self.exclude_ratio = exclude_ratio

        # Couleurs des équipes détectées
        self.team_a_colors: Optional[TeamColors] = None
        self.team_b_colors: Optional[TeamColors] = None
        self.referee_colors: Optional[TeamColors] = None

        # Historique pour la stabilité (réduit pour permettre les corrections)
        self.classification_history: Dict[int, List[str]] = {}
        self.history_length = 5  # Réduit de 10 à 5 pour corrections plus rapides

        # Track IDs confirmés comme arbitres (verrouillés)
        self.confirmed_referees: set = set()
        self.referee_confirmation_threshold = 3  # 3 frames consécutives = arbitre confirmé

        # Calibré
        self.is_calibrated = False

    def _is_referee_color(self, bgr_color: Tuple[int, int, int]) -> bool:
        """
        Vérifier si une couleur BGR correspond à une couleur typique d'arbitre

        Les arbitres portent généralement des couleurs vives:
        - Orange
        - Jaune fluo
        - Rose fluo

        Args:
            bgr_color: Couleur en BGR

        Returns:
            True si c'est une couleur d'arbitre
        """
        # Convertir BGR -> HSV
        bgr = np.uint8([[bgr_color]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

        # Vérifier chaque plage de couleur d'arbitre
        for h_min, h_max, s_min, v_min in self.REFEREE_HSV_RANGES:
            if h_min <= h <= h_max and s >= s_min and v >= v_min:
                return True

        return False

    def _get_color_saturation_value(self, bgr_color: Tuple[int, int, int]) -> Tuple[int, int]:
        """
        Obtenir la saturation et la valeur (luminosité) d'une couleur

        Args:
            bgr_color: Couleur en BGR

        Returns:
            (saturation, value) en HSV
        """
        bgr = np.uint8([[bgr_color]])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        return int(hsv[1]), int(hsv[2])

    def extract_jersey_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Extraire la région du maillot d'une bounding box

        Args:
            frame: Image BGR
            bbox: Bounding box (x1, y1, x2, y2)

        Returns:
            Région du maillot
        """
        x1, y1, x2, y2 = map(int, bbox)

        # Contraindre aux dimensions de l'image
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Extraire le ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return np.array([])

        roi_h, roi_w = roi.shape[:2]

        # Exclure le haut (tête) et le bas (jambes/pieds)
        top_exclude = int(roi_h * self.exclude_ratio)
        bottom_exclude = int(roi_h * (1 - self.exclude_ratio * 1.5))

        # Zone centrale = maillot
        jersey_region = roi[top_exclude:bottom_exclude, :]

        return jersey_region

    def extract_dominant_colors(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        n_colors: int = 2
    ) -> List[Tuple[int, int, int]]:
        """
        Extraire les couleurs dominantes d'une région

        Args:
            frame: Image BGR
            bbox: Bounding box
            n_colors: Nombre de couleurs à extraire

        Returns:
            Liste des couleurs dominantes (BGR)
        """
        jersey = self.extract_jersey_region(frame, bbox)
        if jersey.size == 0:
            return [(128, 128, 128)]

        # Convertir en liste de pixels
        pixels = jersey.reshape(-1, 3)

        # Échantillonner si trop de pixels
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]

        if len(pixels) < n_colors:
            return [tuple(np.mean(pixels, axis=0).astype(int))]

        # K-means clustering
        kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)

        # Trier par fréquence
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_indices = np.argsort(-counts)

        colors = []
        for idx in sorted_indices:
            color = kmeans.cluster_centers_[labels[idx]].astype(int)
            colors.append(tuple(color))

        return colors

    def calibrate(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> bool:
        """
        Calibrer les couleurs des équipes à partir des premières détections

        Sépare d'abord les arbitres (couleurs vives) des joueurs avant
        de faire le clustering des équipes.

        Args:
            frame: Image BGR
            detections: Liste des détections de joueurs

        Returns:
            True si calibration réussie
        """
        if len(detections) < 4:
            return False

        # Extraire les couleurs de tous les joueurs et séparer les arbitres
        team_colors_list = []
        referee_colors_list = []

        for det in detections:
            colors = self.extract_dominant_colors(frame, det.bbox, n_colors=1)
            if colors:
                color = colors[0]
                # Séparer les arbitres des joueurs
                if self._is_referee_color(color):
                    referee_colors_list.append(color)
                else:
                    team_colors_list.append(color)

        # Si on a trouvé des arbitres, calculer leur couleur moyenne
        if referee_colors_list:
            referee_avg = tuple(np.mean(referee_colors_list, axis=0).astype(int))
            self.referee_colors = TeamColors(primary=referee_avg, name="referee")

        # Maintenant, classifier les joueurs (non-arbitres) en 2 équipes
        if len(team_colors_list) < 4:
            # Pas assez de joueurs non-arbitres
            if len(team_colors_list) >= 2:
                # Essayer quand même avec moins de joueurs
                pass
            else:
                return False

        # Clustering des couleurs des joueurs (pas les arbitres)
        colors_array = np.array(team_colors_list)
        n_clusters = min(2, len(team_colors_list))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(colors_array)

        # Trier les clusters par taille
        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        sorted_indices = np.argsort(-counts)

        # Assigner les équipes
        team_colors = []
        for idx in sorted_indices:
            center = kmeans.cluster_centers_[labels[idx]].astype(int)
            team_colors.append(tuple(center))

        if len(team_colors) >= 1:
            self.team_a_colors = TeamColors(primary=team_colors[0], name="team_a")

        if len(team_colors) >= 2:
            self.team_b_colors = TeamColors(primary=team_colors[1], name="team_b")

        self.is_calibrated = True
        return True

    def classify_team(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        track_id: Optional[int] = None
    ) -> str:
        """
        Classifier un joueur dans une équipe

        Vérifie d'abord si c'est un arbitre (couleur orange/jaune/fluo)
        avant de comparer aux couleurs des équipes.

        Args:
            frame: Image BGR
            bbox: Bounding box du joueur
            track_id: ID de tracking pour la stabilité

        Returns:
            "team_a", "team_b", "referee", ou "unknown"
        """
        # PRIORITÉ 0: Si track_id est un arbitre confirmé, retourner directement "referee"
        if track_id is not None and track_id in self.confirmed_referees:
            return "referee"

        # Extraire les couleurs dominantes
        colors = self.extract_dominant_colors(frame, bbox, n_colors=1)
        if not colors:
            return "unknown"

        player_color = colors[0]

        # PRIORITÉ 1: Vérifier si c'est une couleur d'arbitre (orange, jaune fluo, etc.)
        if self._is_referee_color(player_color):
            classification = "referee"
            if track_id is not None:
                classification = self._smooth_classification(track_id, classification)
                # Vérifier si on doit confirmer cet arbitre
                self._check_referee_confirmation(track_id)
            return classification

        if not self.is_calibrated:
            return "unknown"

        # PRIORITÉ 2: Comparer aux couleurs des équipes calibrées
        distances = {}

        if self.team_a_colors:
            # Vérifier que la couleur de l'équipe n'est pas une couleur d'arbitre
            if not self._is_referee_color(self.team_a_colors.primary):
                distances["team_a"] = self._color_distance(
                    player_color, self.team_a_colors.primary
                )

        if self.team_b_colors:
            if not self._is_referee_color(self.team_b_colors.primary):
                distances["team_b"] = self._color_distance(
                    player_color, self.team_b_colors.primary
                )

        # Ne pas utiliser referee_colors pour la distance car on a déjà vérifié au-dessus
        # Si on arrive ici, ce n'est pas un arbitre

        if not distances:
            return "unknown"

        # Trouver la classe la plus proche
        classification = min(distances, key=distances.get)

        # Appliquer un lissage temporel si track_id fourni
        if track_id is not None:
            classification = self._smooth_classification(track_id, classification)

        return classification

    def classify_batch(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        track_ids: Optional[List[int]] = None
    ) -> List[str]:
        """
        Classifier plusieurs joueurs en batch

        Args:
            frame: Image BGR
            detections: Liste des détections
            track_ids: Liste des IDs de tracking (optionnel)

        Returns:
            Liste des classifications
        """
        classifications = []
        for i, det in enumerate(detections):
            track_id = track_ids[i] if track_ids else None
            team = self.classify_team(frame, det.bbox, track_id)
            classifications.append(team)
        return classifications

    def _color_distance(
        self,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int]
    ) -> float:
        """Calculer la distance entre deux couleurs"""
        # Convertir en LAB pour une meilleure perception
        c1 = np.uint8([[color1]])
        c2 = np.uint8([[color2]])

        lab1 = cv2.cvtColor(c1, cv2.COLOR_BGR2LAB)[0][0].astype(float)
        lab2 = cv2.cvtColor(c2, cv2.COLOR_BGR2LAB)[0][0].astype(float)

        # Distance euclidienne en espace LAB
        return np.sqrt(np.sum((lab1 - lab2) ** 2))

    def _smooth_classification(self, track_id: int, classification: str) -> str:
        """Lisser les classifications avec l'historique"""
        if track_id not in self.classification_history:
            self.classification_history[track_id] = []

        history = self.classification_history[track_id]
        history.append(classification)

        # Limiter la taille de l'historique
        if len(history) > self.history_length:
            history.pop(0)

        # Retourner la classification la plus fréquente
        if len(history) >= 3:
            counter = Counter(history)
            return counter.most_common(1)[0][0]

        return classification

    def _check_referee_confirmation(self, track_id: int):
        """
        Vérifier si un track_id doit être confirmé comme arbitre

        Après N classifications consécutives comme referee, le track_id
        est verrouillé comme arbitre pour éviter le clignotement.
        """
        if track_id in self.confirmed_referees:
            return

        history = self.classification_history.get(track_id, [])

        # Compter les classifications referee récentes
        recent_referee_count = 0
        for cls in reversed(history):
            if cls == "referee":
                recent_referee_count += 1
            else:
                break

        # Si assez de classifications consécutives, confirmer l'arbitre
        if recent_referee_count >= self.referee_confirmation_threshold:
            self.confirmed_referees.add(track_id)

    def get_team_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Obtenir les couleurs des équipes"""
        colors = {}
        if self.team_a_colors:
            colors["team_a"] = self.team_a_colors.primary
        if self.team_b_colors:
            colors["team_b"] = self.team_b_colors.primary
        if self.referee_colors:
            colors["referee"] = self.referee_colors.primary
        return colors

    def set_team_colors(
        self,
        team_a: Tuple[int, int, int],
        team_b: Tuple[int, int, int],
        referee: Optional[Tuple[int, int, int]] = None
    ):
        """
        Définir manuellement les couleurs des équipes

        Args:
            team_a: Couleur de l'équipe A (BGR)
            team_b: Couleur de l'équipe B (BGR)
            referee: Couleur de l'arbitre (BGR, optionnel)
        """
        self.team_a_colors = TeamColors(primary=team_a, name="team_a")
        self.team_b_colors = TeamColors(primary=team_b, name="team_b")
        if referee:
            self.referee_colors = TeamColors(primary=referee, name="referee")
        self.is_calibrated = True

    def reset(self):
        """Réinitialiser le classificateur"""
        self.team_a_colors = None
        self.team_b_colors = None
        self.referee_colors = None
        self.classification_history.clear()
        self.confirmed_referees.clear()
        self.is_calibrated = False


class GoalkeeperDetector:
    """Détecteur de gardien de but basé sur la couleur"""

    def __init__(self, team_classifier: TeamClassifier):
        """
        Args:
            team_classifier: Instance du classificateur d'équipes
        """
        self.team_classifier = team_classifier
        self.goalkeeper_colors: Dict[str, Tuple[int, int, int]] = {}

    def detect_goalkeepers(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        pitch_regions: Dict[str, Tuple[int, int, int, int]] = None
    ) -> List[int]:
        """
        Détecter les gardiens de but

        Args:
            frame: Image BGR
            detections: Liste des détections
            pitch_regions: Régions des buts (optionnel)

        Returns:
            Indices des gardiens dans la liste de détections
        """
        goalkeeper_indices = []

        for i, det in enumerate(detections):
            colors = self.team_classifier.extract_dominant_colors(
                frame, det.bbox, n_colors=1
            )
            if not colors:
                continue

            player_color = colors[0]

            # Vérifier si la couleur est différente des couleurs d'équipe
            if self.team_classifier.is_calibrated:
                dist_a = self.team_classifier._color_distance(
                    player_color,
                    self.team_classifier.team_a_colors.primary
                )
                dist_b = self.team_classifier._color_distance(
                    player_color,
                    self.team_classifier.team_b_colors.primary
                )

                # Si la couleur est très différente des deux équipes, c'est peut-être un gardien
                min_dist = min(dist_a, dist_b)
                if min_dist > 80:  # Seuil de distance
                    goalkeeper_indices.append(i)

        return goalkeeper_indices
