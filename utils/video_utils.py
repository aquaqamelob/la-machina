"""
Football Analysis - Utilitaires vidéo
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Generator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoInfo:
    """Informations sur une vidéo"""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float  # en secondes
    codec: str


def get_video_info(video_path: str) -> Optional[VideoInfo]:
    """
    Obtenir les informations d'une vidéo

    Args:
        video_path: Chemin vers la vidéo

    Returns:
        VideoInfo ou None si erreur
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

    cap.release()

    duration = frame_count / fps if fps > 0 else 0

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration=duration,
        codec=codec
    )


def resize_frame(
    frame: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    keep_aspect_ratio: bool = True
) -> np.ndarray:
    """
    Redimensionner un frame

    Args:
        frame: Image à redimensionner
        width: Largeur cible (optionnel)
        height: Hauteur cible (optionnel)
        keep_aspect_ratio: Conserver le ratio d'aspect

    Returns:
        Image redimensionnée
    """
    h, w = frame.shape[:2]

    if width is None and height is None:
        return frame

    if keep_aspect_ratio:
        if width is not None and height is not None:
            # Ajuster pour tenir dans les dimensions
            scale = min(width / w, height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
        elif width is not None:
            scale = width / w
            new_w = width
            new_h = int(h * scale)
        else:
            scale = height / h
            new_w = int(w * scale)
            new_h = height
    else:
        new_w = width or w
        new_h = height or h

    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def create_video_writer(
    output_path: str,
    width: int,
    height: int,
    fps: float,
    codec: str = "mp4v"
) -> cv2.VideoWriter:
    """
    Créer un VideoWriter pour l'écriture vidéo

    Args:
        output_path: Chemin du fichier de sortie
        width: Largeur de la vidéo
        height: Hauteur de la vidéo
        fps: Images par seconde
        codec: Codec vidéo (mp4v, avc1, XVID, etc.)

    Returns:
        VideoWriter configuré
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def frame_generator(
    video_path: str,
    skip_frames: int = 0,
    max_frames: Optional[int] = None,
    start_frame: int = 0
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Générateur de frames pour une vidéo

    Args:
        video_path: Chemin vers la vidéo
        skip_frames: Nombre de frames à sauter entre chaque lecture
        max_frames: Nombre maximum de frames à lire
        start_frame: Frame de départ

    Yields:
        Tuple (numéro_frame, frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")

    # Aller au frame de départ
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    frames_read = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames is not None and frames_read >= max_frames:
                break

            yield frame_idx, frame
            frames_read += 1
            frame_idx += 1

            # Sauter des frames si nécessaire
            if skip_frames > 0:
                for _ in range(skip_frames):
                    ret = cap.grab()
                    if not ret:
                        return
                    frame_idx += 1
    finally:
        cap.release()


def extract_frame(video_path: str, frame_number: int) -> Optional[np.ndarray]:
    """
    Extraire un frame spécifique d'une vidéo

    Args:
        video_path: Chemin vers la vidéo
        frame_number: Numéro du frame à extraire

    Returns:
        Frame ou None si erreur
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def extract_frames_at_times(
    video_path: str,
    times: list,
    fps: Optional[float] = None
) -> list:
    """
    Extraire des frames à des temps spécifiques

    Args:
        video_path: Chemin vers la vidéo
        times: Liste des temps en secondes
        fps: FPS de la vidéo (calculé si non fourni)

    Returns:
        Liste des frames extraits
    """
    if fps is None:
        info = get_video_info(video_path)
        if info is None:
            return []
        fps = info.fps

    frames = []
    for t in times:
        frame_number = int(t * fps)
        frame = extract_frame(video_path, frame_number)
        if frame is not None:
            frames.append(frame)

    return frames


def concatenate_frames_horizontal(frames: list, gap: int = 0) -> np.ndarray:
    """
    Concaténer des frames horizontalement

    Args:
        frames: Liste de frames
        gap: Espace entre les frames

    Returns:
        Image concaténée
    """
    if not frames:
        return np.array([])

    # Redimensionner à la même hauteur
    max_height = max(f.shape[0] for f in frames)
    resized = []
    for frame in frames:
        if frame.shape[0] != max_height:
            scale = max_height / frame.shape[0]
            new_width = int(frame.shape[1] * scale)
            frame = cv2.resize(frame, (new_width, max_height))
        resized.append(frame)

    if gap > 0:
        gap_array = np.zeros((max_height, gap, 3), dtype=np.uint8)
        result = [resized[0]]
        for frame in resized[1:]:
            result.append(gap_array)
            result.append(frame)
        return np.hstack(result)

    return np.hstack(resized)


def concatenate_frames_vertical(frames: list, gap: int = 0) -> np.ndarray:
    """
    Concaténer des frames verticalement

    Args:
        frames: Liste de frames
        gap: Espace entre les frames

    Returns:
        Image concaténée
    """
    if not frames:
        return np.array([])

    # Redimensionner à la même largeur
    max_width = max(f.shape[1] for f in frames)
    resized = []
    for frame in frames:
        if frame.shape[1] != max_width:
            scale = max_width / frame.shape[1]
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (max_width, new_height))
        resized.append(frame)

    if gap > 0:
        gap_array = np.zeros((gap, max_width, 3), dtype=np.uint8)
        result = [resized[0]]
        for frame in resized[1:]:
            result.append(gap_array)
            result.append(frame)
        return np.vstack(result)

    return np.vstack(resized)


def add_text_overlay(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    background: bool = True,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_padding: int = 5
) -> np.ndarray:
    """
    Ajouter du texte avec fond optionnel

    Args:
        frame: Image
        text: Texte à afficher
        position: Position (x, y)
        font_scale: Échelle de la police
        color: Couleur du texte
        thickness: Épaisseur du texte
        background: Ajouter un fond
        bg_color: Couleur du fond
        bg_padding: Padding du fond

    Returns:
        Image avec texte
    """
    result = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculer la taille du texte
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    x, y = position

    if background:
        # Dessiner le fond
        cv2.rectangle(
            result,
            (x - bg_padding, y - text_height - bg_padding),
            (x + text_width + bg_padding, y + baseline + bg_padding),
            bg_color,
            -1
        )

    # Dessiner le texte
    cv2.putText(result, text, (x, y), font, font_scale, color, thickness)

    return result


def create_progress_bar(
    width: int,
    height: int,
    progress: float,
    bg_color: Tuple[int, int, int] = (50, 50, 50),
    bar_color: Tuple[int, int, int] = (0, 200, 0)
) -> np.ndarray:
    """
    Créer une barre de progression

    Args:
        width: Largeur de la barre
        height: Hauteur de la barre
        progress: Progression (0.0 à 1.0)
        bg_color: Couleur du fond
        bar_color: Couleur de la barre

    Returns:
        Image de la barre de progression
    """
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    bar[:, :] = bg_color

    progress_width = int(width * min(1.0, max(0.0, progress)))
    if progress_width > 0:
        bar[:, :progress_width] = bar_color

    return bar


def download_youtube_video(
    url: str,
    output_path: str,
    max_duration: Optional[int] = None
) -> Optional[str]:
    """
    Télécharger une vidéo YouTube

    Args:
        url: URL de la vidéo YouTube
        output_path: Dossier de sortie
        max_duration: Durée maximum en secondes (optionnel)

    Returns:
        Chemin du fichier téléchargé ou None si erreur
    """
    try:
        import yt_dlp

        output_template = str(Path(output_path) / "%(title)s.%(ext)s")

        ydl_opts = {
            'format': 'best[ext=mp4]/best',
            'outtmpl': output_template,
            'quiet': True,
            'no_warnings': True,
        }

        if max_duration:
            ydl_opts['match_filter'] = lambda info: (
                "Video too long" if info.get('duration', 0) > max_duration else None
            )

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)

    except ImportError:
        print("yt-dlp non installé. Installez-le avec: pip install yt-dlp")
        return None
    except Exception as e:
        print(f"Erreur lors du téléchargement: {e}")
        return None
