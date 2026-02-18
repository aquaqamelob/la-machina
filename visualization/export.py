"""
Football Analysis - Export vidéo et rapports
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils.video_utils import create_video_writer, get_video_info


class VideoExporter:
    """
    Exporteur de vidéos analysées et de rapports
    """

    def __init__(self, output_dir: str):
        """
        Initialiser l'exporteur

        Args:
            output_dir: Dossier de sortie
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_video(
        self,
        frames: List[np.ndarray],
        filename: str,
        fps: float = 30.0,
        codec: str = "mp4v"
    ) -> str:
        """
        Exporter une liste de frames en vidéo

        Args:
            frames: Liste d'images
            filename: Nom du fichier de sortie
            fps: FPS de la vidéo
            codec: Codec vidéo

        Returns:
            Chemin du fichier créé
        """
        if not frames:
            raise ValueError("Aucun frame à exporter")

        output_path = self.output_dir / filename
        h, w = frames[0].shape[:2]

        writer = create_video_writer(str(output_path), w, h, fps, codec)

        for frame in frames:
            writer.write(frame)

        writer.release()

        return str(output_path)

    def export_highlights(
        self,
        video_path: str,
        timestamps: List[float],
        duration: float = 5.0,
        filename_prefix: str = "highlight"
    ) -> List[str]:
        """
        Exporter des clips highlight

        Args:
            video_path: Chemin de la vidéo source
            timestamps: Temps centraux des highlights (secondes)
            duration: Durée de chaque clip
            filename_prefix: Préfixe des fichiers

        Returns:
            Liste des chemins créés
        """
        from src.video_processor import VideoClipExtractor

        extractor = VideoClipExtractor(video_path)
        clips = []

        half_duration = duration / 2

        for i, t in enumerate(timestamps):
            start = max(0, t - half_duration)
            end = t + half_duration

            output_path = self.output_dir / f"{filename_prefix}_{i:03d}.mp4"
            if extractor.extract_clip(str(output_path), start, end):
                clips.append(str(output_path))

        return clips

    def export_gif(
        self,
        frames: List[np.ndarray],
        filename: str,
        fps: int = 10,
        scale: float = 0.5
    ) -> str:
        """
        Exporter des frames en GIF animé

        Args:
            frames: Liste d'images
            filename: Nom du fichier de sortie
            fps: FPS du GIF
            scale: Facteur d'échelle

        Returns:
            Chemin du fichier créé
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow est nécessaire pour l'export GIF")

        output_path = self.output_dir / filename

        # Convertir et redimensionner les frames
        pil_frames = []
        for frame in frames:
            # BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Redimensionner
            if scale != 1.0:
                h, w = rgb.shape[:2]
                new_size = (int(w * scale), int(h * scale))
                rgb = cv2.resize(rgb, new_size)

            pil_frames.append(Image.fromarray(rgb))

        # Sauvegarder le GIF
        duration = int(1000 / fps)  # millisecondes par frame
        pil_frames[0].save(
            str(output_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )

        return str(output_path)

    def export_heatmap_overlay(
        self,
        background: np.ndarray,
        heatmap: np.ndarray,
        filename: str,
        alpha: float = 0.6,
        colormap: str = "jet"
    ) -> str:
        """
        Exporter une heatmap superposée à une image

        Args:
            background: Image de fond
            heatmap: Grille de heatmap (0-1)
            filename: Nom du fichier
            alpha: Transparence
            colormap: Colormap matplotlib

        Returns:
            Chemin du fichier créé
        """
        import matplotlib.pyplot as plt

        output_path = self.output_dir / filename

        # Appliquer le colormap
        cmap = plt.get_cmap(colormap)
        colored = cmap(heatmap)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)
        colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

        # Redimensionner à la taille du fond
        h, w = background.shape[:2]
        colored = cv2.resize(colored, (w, h))

        # Créer le masque
        mask = cv2.resize(heatmap, (w, h))
        mask = (mask * alpha).clip(0, 1)
        mask = np.stack([mask] * 3, axis=-1)

        # Fusionner
        result = (
            background.astype(np.float32) * (1 - mask) +
            colored.astype(np.float32) * mask
        ).astype(np.uint8)

        cv2.imwrite(str(output_path), result)
        return str(output_path)

    def export_stats_json(
        self,
        stats: Dict,
        filename: str = "match_stats.json"
    ) -> str:
        """
        Exporter les statistiques en JSON

        Args:
            stats: Dictionnaire de statistiques
            filename: Nom du fichier

        Returns:
            Chemin du fichier créé
        """
        output_path = self.output_dir / filename

        # Ajouter des métadonnées
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "stats": stats
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def export_stats_csv(
        self,
        player_stats: Dict,
        filename: str = "player_stats.csv"
    ) -> str:
        """
        Exporter les statistiques des joueurs en CSV

        Args:
            player_stats: Statistiques par joueur
            filename: Nom du fichier

        Returns:
            Chemin du fichier créé
        """
        import pandas as pd

        output_path = self.output_dir / filename

        # Convertir en DataFrame
        rows = []
        for player_id, stats in player_stats.items():
            row = {"player_id": player_id}
            if hasattr(stats, '__dict__'):
                row.update(vars(stats))
            else:
                row.update(stats)
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        return str(output_path)

    def export_report_html(
        self,
        stats: Dict,
        heatmap_paths: Dict[str, str],
        filename: str = "match_report.html"
    ) -> str:
        """
        Exporter un rapport HTML complet

        Args:
            stats: Statistiques du match
            heatmap_paths: Chemins des heatmaps
            filename: Nom du fichier

        Returns:
            Chemin du fichier créé
        """
        output_path = self.output_dir / filename

        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport d'analyse - Match de Football</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #3498db, #2ecc71);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-item {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .team-comparison {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .team-a {{ border-left: 4px solid #3498db; }}
        .team-b {{ border-left: 4px solid #e74c3c; }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{ background: #f8f9fa; }}
        .heatmap-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .heatmap-container img {{
            width: 100%;
            border-radius: 8px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>⚽ Rapport d'Analyse de Match</h1>
        <p>Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
    </div>

    <div class="card">
        <h2>📊 Statistiques Générales</h2>
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">{stats.get('duration', 0):.1f}s</div>
                <div class="stat-label">Durée analysée</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats.get('frames_analyzed', 0)}</div>
                <div class="stat-label">Frames traités</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats.get('ball_distance', 0):.0f}m</div>
                <div class="stat-label">Distance ballon</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{stats.get('possession_changes', 0)}</div>
                <div class="stat-label">Changements possession</div>
            </div>
        </div>
    </div>

    <div class="team-comparison">
        <div class="card team-a">
            <h2>🔵 Équipe A</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{stats.get('team_a', {}).get('possession', 0):.1f}%</div>
                    <div class="stat-label">Possession</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('team_a', {}).get('total_distance', 0):.0f}m</div>
                    <div class="stat-label">Distance</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('team_a', {}).get('sprints', 0)}</div>
                    <div class="stat-label">Sprints</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('team_a', {}).get('max_speed', 0):.1f}</div>
                    <div class="stat-label">Vitesse max (km/h)</div>
                </div>
            </div>
        </div>

        <div class="card team-b">
            <h2>🔴 Équipe B</h2>
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{stats.get('team_b', {}).get('possession', 0):.1f}%</div>
                    <div class="stat-label">Possession</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('team_b', {}).get('total_distance', 0):.0f}m</div>
                    <div class="stat-label">Distance</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('team_b', {}).get('sprints', 0)}</div>
                    <div class="stat-label">Sprints</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{stats.get('team_b', {}).get('max_speed', 0):.1f}</div>
                    <div class="stat-label">Vitesse max (km/h)</div>
                </div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>🔥 Heatmaps</h2>
        <div class="heatmap-container">
            {''.join(f'<div><h3>{name}</h3><img src="{path}" alt="{name}"></div>' for name, path in heatmap_paths.items())}
        </div>
    </div>

    <div class="card">
        <h2>👥 Statistiques par Joueur</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Équipe</th>
                    <th>Distance (m)</th>
                    <th>Vitesse max (km/h)</th>
                    <th>Vitesse moy (km/h)</th>
                    <th>Sprints</th>
                </tr>
            </thead>
            <tbody>
                {''.join(f'''
                <tr>
                    <td>{pid}</td>
                    <td>{data.get('team', 'N/A')}</td>
                    <td>{data.get('distance', 0):.1f}</td>
                    <td>{data.get('max_speed', 0):.1f}</td>
                    <td>{data.get('avg_speed', 0):.1f}</td>
                    <td>{data.get('sprints', 0)}</td>
                </tr>
                ''' for pid, data in stats.get('players', {}).items())}
            </tbody>
        </table>
    </div>

    <footer style="text-align: center; color: #7f8c8d; margin-top: 40px;">
        <p>Généré par Football Match Analysis avec YOLO</p>
    </footer>
</body>
</html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return str(output_path)

    def create_thumbnail(
        self,
        video_path: str,
        filename: str = "thumbnail.jpg",
        frame_position: float = 0.5
    ) -> str:
        """
        Créer une miniature de la vidéo

        Args:
            video_path: Chemin de la vidéo
            filename: Nom du fichier
            frame_position: Position relative du frame (0-1)

        Returns:
            Chemin du fichier créé
        """
        output_path = self.output_dir / filename

        info = get_video_info(video_path)
        if not info:
            raise ValueError(f"Impossible de lire la vidéo: {video_path}")

        frame_idx = int(info.frame_count * frame_position)

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(str(output_path), frame)
            return str(output_path)

        raise RuntimeError("Impossible d'extraire le frame")
