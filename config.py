"""
Football Match Analysis - Configuration centralisée
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import os


def get_device() -> str:
    """Détecter le meilleur device disponible (CUDA, MPS pour Mac, ou CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"  # Mac Apple Silicon (M1/M2/M3)
        else:
            return "cpu"
    except ImportError:
        return "cpu"


@dataclass
class ModelConfig:
    """Configuration des modèles YOLO"""
    # Modèles de détection (YOLO26 - janvier 2026)
    detection_model: str = "yolo26m.pt"  # ou yolo26n.pt pour temps réel
    pose_model: str = "yolo26m-pose.pt"  # ou yolo26n-pose.pt pour temps réel

    # Seuils de détection (abaissés pour détecter plus de joueurs)
    detection_confidence: float = 0.3  # Réduit de 0.5 pour plus de détections
    ball_confidence: float = 0.2  # Seuil spécifique pour le ballon (plus petit)
    pose_confidence: float = 0.4
    iou_threshold: float = 0.5  # Augmenté pour éviter doublons

    # Classes COCO à détecter
    person_class_id: int = 0  # "person" dans COCO
    ball_class_id: int = 32   # "sports ball" dans COCO

    # Device - détecté automatiquement (cuda, mps, ou cpu)
    device: str = field(default_factory=get_device)


@dataclass
class PitchConfig:
    """Configuration du terrain de football"""
    # Dimensions officielles du terrain (en mètres)
    length: float = 105.0  # longueur
    width: float = 68.0    # largeur

    # Zones importantes
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0

    # Dimensions pour la visualisation 2D (pixels)
    viz_width: int = 1050
    viz_height: int = 680
    viz_margin: int = 50

    @property
    def pixels_per_meter_x(self) -> float:
        return self.viz_width / self.length

    @property
    def pixels_per_meter_y(self) -> float:
        return self.viz_height / self.width


@dataclass
class TrackingConfig:
    """Configuration du tracking"""
    # ByteTrack parameters
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8

    # Historique des positions
    max_history_length: int = 90  # 3 secondes à 30 FPS
    min_hits: int = 3  # Nombre minimum de détections avant confirmation
    max_age: int = 30  # Frames avant suppression d'un track perdu

    # Re-identification
    reid_threshold: float = 0.7


@dataclass
class VideoConfig:
    """Configuration vidéo"""
    # FPS cible pour l'analyse
    target_fps: int = 30

    # Résolution de sortie
    output_width: int = 1920
    output_height: int = 1080

    # Codec vidéo
    fourcc: str = "mp4v"
    output_extension: str = ".mp4"

    # Processing
    skip_frames: int = 0  # 0 = traiter tous les frames
    max_frames: Optional[int] = None  # None = traiter toute la vidéo


@dataclass
class TeamColors:
    """Couleurs des équipes pour la visualisation"""
    # Couleurs principales (BGR pour OpenCV)
    team_a: Tuple[int, int, int] = (219, 152, 52)   # #3498db bleu
    team_b: Tuple[int, int, int] = (60, 76, 231)    # #e74c3c rouge
    referee: Tuple[int, int, int] = (113, 204, 46)  # #2ecc71 vert
    ball: Tuple[int, int, int] = (0, 196, 241)      # #f1c40f jaune
    unknown: Tuple[int, int, int] = (128, 128, 128) # gris

    # Couleurs pour l'overlay
    overlay_bg: Tuple[int, int, int, int] = (0, 0, 0, 178)  # rgba(0,0,0,0.7)
    text_primary: Tuple[int, int, int] = (255, 255, 255)
    text_secondary: Tuple[int, int, int] = (200, 200, 200)


@dataclass
class SpeedConfig:
    """Configuration pour le calcul de vitesse"""
    # Conversion
    pixels_per_meter: float = 10.0  # À calibrer selon la vidéo

    # Lissage
    smoothing_window: int = 5  # Moyenne mobile sur 5 frames

    # Seuils de vitesse (km/h)
    walking_threshold: float = 7.0
    jogging_threshold: float = 14.0
    running_threshold: float = 21.0
    sprint_threshold: float = 25.0
    max_realistic_speed: float = 40.0  # Vitesse max réaliste


@dataclass
class HeatmapConfig:
    """Configuration des heatmaps"""
    # Résolution de la grille
    grid_width: int = 105
    grid_height: int = 68

    # Paramètres du blur gaussien
    blur_sigma: float = 3.0

    # Colormap
    colormap: str = "jet"
    alpha: float = 0.6  # Transparence de la heatmap


@dataclass
class VisualizationConfig:
    """Configuration de la visualisation"""
    # Épaisseurs de ligne
    bbox_thickness: int = 2
    skeleton_thickness: int = 2
    trajectory_thickness: int = 2

    # Tailles de police
    font_scale: float = 0.6
    font_thickness: int = 2

    # Minimap
    minimap_width: int = 300
    minimap_height: int = 195
    minimap_margin: int = 20
    minimap_position: str = "bottom_right"  # bottom_right, bottom_left, top_right, top_left

    # Stats panel
    stats_panel_width: int = 250
    stats_panel_height: int = 150
    stats_panel_position: str = "top_left"

    # Trajectoire
    trajectory_length: int = 30  # Nombre de points à afficher
    trajectory_fade: bool = True  # Effet de fondu

    # Skeleton
    draw_skeleton: bool = True
    skeleton_style: str = "elegant"  # elegant, simple, detailed


@dataclass
class Config:
    """Configuration globale du projet"""
    # Chemins
    project_root: Path = field(default_factory=lambda: Path(__file__).parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent / "models")
    input_dir: Path = field(default_factory=lambda: Path(__file__).parent / "input")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent / "output")

    # Sous-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    colors: TeamColors = field(default_factory=TeamColors)
    speed: SpeedConfig = field(default_factory=SpeedConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        """Créer les dossiers nécessaires"""
        self.models_dir.mkdir(exist_ok=True)
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    @classmethod
    def load_from_yaml(cls, path: str) -> "Config":
        """Charger la configuration depuis un fichier YAML"""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()

    def save_to_yaml(self, path: str):
        """Sauvegarder la configuration vers un fichier YAML"""
        import yaml
        from dataclasses import asdict
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


# Instance globale de configuration
config = Config()


# COCO Keypoints indices pour pose estimation
COCO_KEYPOINTS = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Connexions du squelette pour le dessin
SKELETON_CONNECTIONS = [
    # Tête
    ('nose', 'left_eye'),
    ('nose', 'right_eye'),
    ('left_eye', 'left_ear'),
    ('right_eye', 'right_ear'),
    # Torse
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    # Bras gauche
    ('left_shoulder', 'left_elbow'),
    ('left_elbow', 'left_wrist'),
    # Bras droit
    ('right_shoulder', 'right_elbow'),
    ('right_elbow', 'right_wrist'),
    # Jambe gauche
    ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'),
    # Jambe droite
    ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle'),
]
