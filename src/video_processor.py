"""
Football Analysis - Traitement vidéo
"""

import cv2
import numpy as np
from typing import Optional, Generator, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils.video_utils import get_video_info, VideoInfo, create_video_writer


@dataclass
class ProcessingStats:
    """Statistiques de traitement"""
    total_frames: int = 0
    processed_frames: int = 0
    fps_processing: float = 0.0
    elapsed_time: float = 0.0


class VideoProcessor:
    """
    Processeur vidéo pour la lecture et l'écriture

    Gère le flux vidéo avec options de configuration.
    """

    def __init__(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        target_fps: Optional[int] = None,
        output_width: Optional[int] = None,
        output_height: Optional[int] = None,
        skip_frames: int = 0,
        max_frames: Optional[int] = None
    ):
        """
        Initialiser le processeur

        Args:
            input_path: Chemin de la vidéo d'entrée
            output_path: Chemin de la vidéo de sortie (optionnel)
            target_fps: FPS cible (None = garder l'original)
            output_width: Largeur de sortie
            output_height: Hauteur de sortie
            skip_frames: Nombre de frames à sauter
            max_frames: Nombre max de frames à traiter
        """
        self.input_path = input_path
        self.output_path = output_path
        self.skip_frames = skip_frames
        self.max_frames = max_frames

        # Obtenir les infos de la vidéo
        self.video_info = get_video_info(input_path)
        if self.video_info is None:
            raise ValueError(f"Impossible d'ouvrir la vidéo: {input_path}")

        # Configuration de sortie
        self.target_fps = target_fps or self.video_info.fps
        self.output_width = output_width or self.video_info.width
        self.output_height = output_height or self.video_info.height

        # État
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_frame = 0
        self.is_open = False

        # Stats
        self.stats = ProcessingStats(total_frames=self.video_info.frame_count)

    def open(self):
        """Ouvrir la vidéo pour le traitement"""
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la vidéo: {self.input_path}")

        if self.output_path:
            self.writer = create_video_writer(
                self.output_path,
                self.output_width,
                self.output_height,
                self.target_fps
            )

        self.is_open = True
        self.current_frame = 0

    def close(self):
        """Fermer les ressources"""
        if self.cap:
            self.cap.release()
            self.cap = None

        if self.writer:
            self.writer.release()
            self.writer = None

        self.is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lire un frame

        Returns:
            (success, frame) ou (False, None) si fin de vidéo
        """
        if not self.is_open:
            return False, None

        ret, frame = self.cap.read()
        if not ret:
            return False, None

        self.current_frame += 1
        self.stats.processed_frames += 1

        # Redimensionner si nécessaire
        if (frame.shape[1] != self.output_width or
            frame.shape[0] != self.output_height):
            frame = cv2.resize(
                frame,
                (self.output_width, self.output_height),
                interpolation=cv2.INTER_LINEAR
            )

        return True, frame

    def write_frame(self, frame: np.ndarray):
        """Écrire un frame"""
        if self.writer:
            # S'assurer que les dimensions sont correctes
            if (frame.shape[1] != self.output_width or
                frame.shape[0] != self.output_height):
                frame = cv2.resize(
                    frame,
                    (self.output_width, self.output_height),
                    interpolation=cv2.INTER_LINEAR
                )
            self.writer.write(frame)

    def frames(
        self,
        with_progress: bool = True,
        description: str = "Processing"
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Générateur de frames avec barre de progression

        Args:
            with_progress: Afficher une barre de progression
            description: Description pour la barre

        Yields:
            (frame_index, frame)
        """
        if not self.is_open:
            self.open()

        total = self.max_frames or self.video_info.frame_count
        if self.skip_frames > 0:
            total = total // (self.skip_frames + 1)

        iterator = range(total)
        if with_progress:
            iterator = tqdm(iterator, desc=description, unit="frame")

        frames_yielded = 0

        for _ in iterator:
            ret, frame = self.read_frame()
            if not ret:
                break

            yield self.current_frame - 1, frame
            frames_yielded += 1

            if self.max_frames and frames_yielded >= self.max_frames:
                break

            # Sauter des frames si configuré
            if self.skip_frames > 0:
                for _ in range(self.skip_frames):
                    ret = self.cap.grab()
                    if not ret:
                        return
                    self.current_frame += 1

    def process(
        self,
        process_fn: Callable[[int, np.ndarray], np.ndarray],
        with_progress: bool = True
    ):
        """
        Traiter la vidéo avec une fonction

        Args:
            process_fn: Fonction (frame_idx, frame) -> processed_frame
            with_progress: Afficher la progression
        """
        import time
        start_time = time.time()

        for frame_idx, frame in self.frames(with_progress):
            # Traiter le frame
            processed = process_fn(frame_idx, frame)

            # Écrire le résultat
            self.write_frame(processed)

        self.stats.elapsed_time = time.time() - start_time
        if self.stats.elapsed_time > 0:
            self.stats.fps_processing = (
                self.stats.processed_frames / self.stats.elapsed_time
            )

    def extract_frames(
        self,
        output_dir: str,
        frame_indices: list = None,
        interval: int = None,
        format: str = "jpg"
    ) -> list:
        """
        Extraire des frames en images

        Args:
            output_dir: Dossier de sortie
            frame_indices: Indices spécifiques à extraire
            interval: Extraire tous les N frames
            format: Format d'image (jpg, png)

        Returns:
            Liste des chemins des images extraites
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        if frame_indices:
            indices_set = set(frame_indices)
        else:
            indices_set = None

        with self:
            for frame_idx, frame in self.frames(with_progress=True, description="Extracting"):
                save = False

                if indices_set and frame_idx in indices_set:
                    save = True
                elif interval and frame_idx % interval == 0:
                    save = True

                if save:
                    filename = f"frame_{frame_idx:06d}.{format}"
                    filepath = output_path / filename
                    cv2.imwrite(str(filepath), frame)
                    saved_paths.append(str(filepath))

        return saved_paths

    def get_frame_at(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Obtenir un frame spécifique

        Args:
            frame_number: Numéro du frame

        Returns:
            Frame ou None
        """
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.input_path)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        return frame if ret else None

    def get_time_at_frame(self, frame_number: int) -> float:
        """Obtenir le temps en secondes pour un numéro de frame"""
        return frame_number / self.video_info.fps

    def get_frame_at_time(self, time_seconds: float) -> int:
        """Obtenir le numéro de frame pour un temps donné"""
        return int(time_seconds * self.video_info.fps)

    @property
    def fps(self) -> float:
        """FPS de la vidéo"""
        return self.video_info.fps

    @property
    def frame_count(self) -> int:
        """Nombre total de frames"""
        return self.video_info.frame_count

    @property
    def duration(self) -> float:
        """Durée en secondes"""
        return self.video_info.duration

    @property
    def width(self) -> int:
        """Largeur de la vidéo"""
        return self.video_info.width

    @property
    def height(self) -> int:
        """Hauteur de la vidéo"""
        return self.video_info.height

    @property
    def progress(self) -> float:
        """Progression du traitement (0-1)"""
        if self.frame_count > 0:
            return self.current_frame / self.frame_count
        return 0.0


class VideoClipExtractor:
    """Extracteur de clips vidéo"""

    def __init__(self, input_path: str):
        """
        Args:
            input_path: Chemin de la vidéo source
        """
        self.input_path = input_path
        self.video_info = get_video_info(input_path)

    def extract_clip(
        self,
        output_path: str,
        start_time: float,
        end_time: float
    ) -> bool:
        """
        Extraire un clip

        Args:
            output_path: Chemin de sortie
            start_time: Temps de début en secondes
            end_time: Temps de fin en secondes

        Returns:
            True si réussi
        """
        cap = cv2.VideoCapture(self.input_path)
        if not cap.isOpened():
            return False

        fps = self.video_info.fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer = create_video_writer(
            output_path,
            self.video_info.width,
            self.video_info.height,
            fps
        )

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        cap.release()
        writer.release()

        return True

    def extract_highlights(
        self,
        output_dir: str,
        timestamps: list,
        duration: float = 5.0
    ) -> list:
        """
        Extraire plusieurs highlights

        Args:
            output_dir: Dossier de sortie
            timestamps: Liste des temps centraux en secondes
            duration: Durée de chaque highlight

        Returns:
            Liste des chemins des clips
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        clips = []
        half_duration = duration / 2

        for i, t in enumerate(timestamps):
            start = max(0, t - half_duration)
            end = min(self.video_info.duration, t + half_duration)

            clip_path = output_path / f"highlight_{i:03d}.mp4"
            if self.extract_clip(str(clip_path), start, end):
                clips.append(str(clip_path))

        return clips
