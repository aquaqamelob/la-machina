#!/usr/bin/env python3
"""
Football Match Analysis - Pipeline principal

Analyse complète de matchs de football avec YOLO.
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from config import config
from src.detector import PlayerBallDetector
from src.pose_estimator import PoseEstimator, match_poses_to_detections
from src.tracker import MultiObjectTracker
from src.team_classifier import TeamClassifier
from src.speed_calculator import SpeedDistanceCalculator
from src.heatmap_generator import HeatmapGenerator
from src.trajectory_drawer import TrajectoryDrawer
from src.pitch_mapper import PitchMapper
from src.stats_analyzer import MatchStatsAnalyzer
from src.video_processor import VideoProcessor
from src.possession_tracker import PossessionTracker
from visualization.overlay import VideoOverlay


class FootballAnalyzer:
    """
    Analyseur principal de matchs de football

    Pipeline complet de la détection à l'export vidéo.
    """

    def __init__(
        self,
        enable_pose: bool = False,
        enable_speed: bool = False,
        enable_heatmap: bool = False,
        enable_minimap: bool = True,
        enable_stats: bool = False,
        calibration_frames: int = 30
    ):
        """
        Initialiser l'analyseur

        Args:
            enable_pose: Activer l'estimation de pose
            enable_speed: Activer le calcul de vitesse (désactivé - YOLO imprécis)
            enable_heatmap: Activer la génération de heatmaps
            enable_minimap: Activer la minimap
            enable_stats: Activer les statistiques (désactivé - mesures imprécises)
            calibration_frames: Nombre de frames pour la calibration
        """
        self.enable_pose = enable_pose
        self.enable_speed = enable_speed
        self.enable_heatmap = enable_heatmap
        self.enable_minimap = enable_minimap
        self.enable_stats = enable_stats
        self.calibration_frames = calibration_frames

        # Composants
        self.detector = None
        self.pose_estimator = None
        self.tracker = None
        self.team_classifier = None
        self.speed_calculator = None
        self.heatmap_generator = None
        self.trajectory_drawer = None
        self.pitch_mapper = None
        self.stats_analyzer = None
        self.possession_tracker = None
        self.overlay = None

        # État
        self.is_calibrated = False
        self.fps = 30.0

    def initialize(self, fps: float = 30.0, frame_width: int = 1920, frame_height: int = 1080):
        """
        Initialiser tous les composants

        Args:
            fps: FPS de la vidéo
            frame_width: Largeur du frame
            frame_height: Hauteur du frame
        """
        self.fps = fps

        print("Initialisation des composants...")

        # Détection
        print("  - Chargement du détecteur YOLO...")
        self.detector = PlayerBallDetector()

        # Pose estimation
        if self.enable_pose:
            print("  - Chargement de l'estimateur de pose...")
            self.pose_estimator = PoseEstimator()

        # Tracking
        print("  - Initialisation du tracker...")
        self.tracker = MultiObjectTracker()

        # Classification d'équipe
        print("  - Initialisation du classificateur d'équipe...")
        self.team_classifier = TeamClassifier()

        # Calcul de vitesse
        if self.enable_speed:
            print("  - Initialisation du calculateur de vitesse...")
            self.speed_calculator = SpeedDistanceCalculator(fps=fps)

        # Heatmap
        if self.enable_heatmap:
            print("  - Initialisation du générateur de heatmap...")
            self.heatmap_generator = HeatmapGenerator()
            self.heatmap_generator.set_video_dimensions(frame_width, frame_height)

        # Trajectoire
        print("  - Initialisation du dessinateur de trajectoires...")
        self.trajectory_drawer = TrajectoryDrawer()

        # Pitch mapper
        if self.enable_minimap:
            print("  - Initialisation du pitch mapper...")
            self.pitch_mapper = PitchMapper()

        # Stats
        if self.enable_stats:
            print("  - Initialisation de l'analyseur de statistiques...")
            self.stats_analyzer = MatchStatsAnalyzer(fps=fps)

        # Possession tracker (toujours actif pour le footer)
        print("  - Initialisation du tracker de possession...")
        self.possession_tracker = PossessionTracker()

        # Overlay
        print("  - Initialisation de l'overlay vidéo...")
        self.overlay = VideoOverlay()

        print("Initialisation terminée!")

    def calibrate(self, frames: list):
        """
        Calibrer les couleurs d'équipe avec les premiers frames

        Args:
            frames: Liste des premiers frames pour la calibration
        """
        print(f"Calibration sur {len(frames)} frames...")

        all_detections = []
        for frame in frames:
            result = self.detector.detect(frame)
            all_detections.extend(result.players)

        # Calibrer les couleurs d'équipe
        if all_detections and len(frames) > 0:
            # Utiliser le frame du milieu pour la calibration
            mid_frame = frames[len(frames) // 2]
            result = self.detector.detect(mid_frame)

            if self.team_classifier.calibrate(mid_frame, result.players):
                print("  - Couleurs d'équipe calibrées")
                colors = self.team_classifier.get_team_colors()
                for team, color in colors.items():
                    print(f"    - {team}: BGR{color}")
                self.is_calibrated = True
            else:
                print("  - Échec de la calibration des couleurs")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int
    ) -> np.ndarray:
        """
        Traiter un seul frame

        Args:
            frame: Image BGR
            frame_idx: Index du frame

        Returns:
            Frame annoté
        """
        # Détection
        detection_result = self.detector.detect(frame)

        # Tracking
        tracking_result = self.tracker.update(detection_result)

        # Pose estimation
        poses = {}
        if self.enable_pose and self.pose_estimator:
            all_poses = self.pose_estimator.estimate_poses(frame)
            poses = match_poses_to_detections(all_poses, detection_result.players)

        # Classification d'équipe et mise à jour des stats
        for player in tracking_result.players:
            # Classification
            player.team = self.team_classifier.classify_team(
                frame, player.bbox, player.track_id
            )

            # Calcul de vitesse
            if self.enable_speed and self.speed_calculator:
                player.speed = self.speed_calculator.update(
                    player.track_id, player.bottom_center
                )

            # Heatmap
            if self.enable_heatmap and self.heatmap_generator:
                self.heatmap_generator.add_position(
                    player.track_id, player.bottom_center, player.team
                )

        # Stats avancées (optionnel)
        ball_pos = tracking_result.ball.center if tracking_result.ball else None
        if self.enable_stats and self.stats_analyzer:
            self.stats_analyzer.update(
                tracking_result.players,
                ball_pos,
                frame_idx
            )

        # Tracker de possession (toujours actif)
        if self.possession_tracker:
            self.possession_tracker.update(
                tracking_result.players,
                tracking_result.ball
            )

        # Obtenir les stats pour le footer
        possession_stats = {}
        if self.possession_tracker:
            possession_stats = self.possession_tracker.get_stats()

        # Dessiner l'overlay
        annotated = self.overlay.draw_all(
            frame,
            tracking_result.players,
            tracking_result.ball,
            poses if self.enable_pose else None,
            possession_stats,  # Stats pour le footer
            self.pitch_mapper if self.enable_minimap else None,
            draw_trajectories=False,
            draw_minimap=self.enable_minimap,
            draw_stats=False,
            draw_skeletons=self.enable_pose,
            draw_footer=True  # Footer avec possession et passes
        )

        return annotated

    def analyze_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
        skip_frames: int = 0
    ):
        """
        Analyser une vidéo complète

        Args:
            input_path: Chemin de la vidéo d'entrée
            output_path: Chemin de la vidéo de sortie
            max_frames: Nombre max de frames à traiter
            skip_frames: Frames à sauter entre chaque traitement
        """
        print(f"\n{'='*60}")
        print(f"ANALYSE VIDÉO: {input_path}")
        print(f"{'='*60}\n")

        # Créer le processeur vidéo
        processor = VideoProcessor(
            input_path,
            output_path,
            max_frames=max_frames,
            skip_frames=skip_frames
        )

        print(f"Vidéo: {processor.width}x{processor.height} @ {processor.fps:.1f} fps")
        print(f"Durée: {processor.duration:.1f}s ({processor.frame_count} frames)")
        print()

        # Initialiser les composants
        self.initialize(
            fps=processor.fps,
            frame_width=processor.width,
            frame_height=processor.height
        )

        # Ouvrir la vidéo
        processor.open()

        # Phase de calibration
        print("\nPhase de calibration...")
        calibration_frames = []
        for _ in range(self.calibration_frames):
            ret, frame = processor.read_frame()
            if not ret:
                break
            calibration_frames.append(frame)

        self.calibrate(calibration_frames)

        # Rembobiner
        processor.close()
        processor.open()

        # Traitement principal
        print("\nTraitement de la vidéo...")

        for frame_idx, frame in processor.frames(with_progress=True):
            # Traiter le frame
            annotated = self.process_frame(frame, frame_idx)

            # Écrire le résultat
            processor.write_frame(annotated)

        # Fermer
        processor.close()

        print(f"\nVidéo de sortie: {output_path}")
        print(f"Frames traités: {processor.stats.processed_frames}")
        if processor.stats.elapsed_time > 0:
            print(f"Vitesse: {processor.stats.fps_processing:.1f} FPS")

        # Générer les sorties supplémentaires
        self._generate_outputs(output_path)

    def _generate_outputs(self, output_base: str):
        """Générer les sorties supplémentaires (heatmaps, stats)"""
        output_dir = Path(output_base).parent

        # Heatmaps
        if self.enable_heatmap and self.heatmap_generator:
            print("\nGénération des heatmaps...")

            # Heatmap globale
            heatmap = self.heatmap_generator.generate_heatmap()
            heatmap_path = output_dir / "heatmap_global.png"
            self.heatmap_generator.save_heatmap(heatmap, str(heatmap_path))
            print(f"  - Heatmap globale: {heatmap_path}")

            # Heatmaps par équipe
            for team in ["team_a", "team_b"]:
                heatmap = self.heatmap_generator.generate_heatmap(team=team)
                heatmap_path = output_dir / f"heatmap_{team}.png"
                self.heatmap_generator.save_heatmap(heatmap, str(heatmap_path))
                print(f"  - Heatmap {team}: {heatmap_path}")

        # Stats
        if self.enable_stats and self.stats_analyzer:
            print("\nExport des statistiques...")
            stats_path = output_dir / "match_stats.json"
            self.stats_analyzer.export_to_json(str(stats_path))
            print(f"  - Statistiques: {stats_path}")

            # Afficher le résumé
            report = self.stats_analyzer.generate_report()
            print(f"\n{'='*40}")
            print("RÉSUMÉ DU MATCH")
            print(f"{'='*40}")
            print(f"Durée analysée: {report.duration:.1f}s")
            print(f"Frames traités: {report.frames_analyzed}")
            if report.team_a_stats:
                print(f"\nÉquipe A:")
                print(f"  - Possession: {report.team_a_stats.possession_percentage:.1f}%")
                print(f"  - Distance totale: {report.team_a_stats.total_distance:.0f}m")
                print(f"  - Sprints: {report.team_a_stats.total_sprints}")
            if report.team_b_stats:
                print(f"\nÉquipe B:")
                print(f"  - Possession: {report.team_b_stats.possession_percentage:.1f}%")
                print(f"  - Distance totale: {report.team_b_stats.total_distance:.0f}m")
                print(f"  - Sprints: {report.team_b_stats.total_sprints}")
            print(f"{'='*40}")


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="Football Match Analysis avec YOLO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python main.py input.mp4 output.mp4
  python main.py input.mp4 output.mp4 --max-frames 1000
  python main.py input.mp4 output.mp4 --no-pose --no-stats
        """
    )

    parser.add_argument("input", help="Chemin de la vidéo d'entrée")
    parser.add_argument("output", help="Chemin de la vidéo de sortie")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Nombre maximum de frames à traiter")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Nombre de frames à sauter entre chaque traitement")
    parser.add_argument("--no-pose", action="store_true",
                        help="Désactiver l'estimation de pose")
    parser.add_argument("--no-speed", action="store_true",
                        help="Désactiver le calcul de vitesse")
    parser.add_argument("--no-heatmap", action="store_true",
                        help="Désactiver la génération de heatmaps")
    parser.add_argument("--no-minimap", action="store_true",
                        help="Désactiver la minimap")
    parser.add_argument("--no-stats", action="store_true",
                        help="Désactiver les statistiques")
    parser.add_argument("--calibration-frames", type=int, default=30,
                        help="Nombre de frames pour la calibration")

    args = parser.parse_args()

    # Vérifier que l'entrée existe
    if not Path(args.input).exists():
        print(f"Erreur: Le fichier d'entrée n'existe pas: {args.input}")
        return 1

    # Créer le dossier de sortie si nécessaire
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Créer l'analyseur
    analyzer = FootballAnalyzer(
        enable_pose=not args.no_pose,
        enable_speed=not args.no_speed,
        enable_heatmap=not args.no_heatmap,
        enable_minimap=not args.no_minimap,
        enable_stats=not args.no_stats,
        calibration_frames=args.calibration_frames
    )

    # Lancer l'analyse
    try:
        analyzer.analyze_video(
            args.input,
            args.output,
            max_frames=args.max_frames,
            skip_frames=args.skip_frames
        )
        print("\nAnalyse terminée avec succès!")
        return 0
    except KeyboardInterrupt:
        print("\nAnalyse interrompue par l'utilisateur.")
        return 1
    except Exception as e:
        print(f"\nErreur lors de l'analyse: {e}")
        raise


if __name__ == "__main__":
    exit(main())
