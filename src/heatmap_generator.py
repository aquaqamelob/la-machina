"""
Football Analysis - Génération de heatmaps
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config


@dataclass
class HeatmapData:
    """Données d'une heatmap"""
    grid: np.ndarray
    player_id: Optional[int] = None
    team: Optional[str] = None
    normalized: bool = False


class HeatmapGenerator:
    """
    Générateur de heatmaps pour visualiser les zones de jeu

    Crée des heatmaps basées sur les positions des joueurs.
    """

    def __init__(
        self,
        width: int = None,
        height: int = None,
        sigma: float = None
    ):
        """
        Initialiser le générateur

        Args:
            width: Largeur de la grille (en pixels ou unités)
            height: Hauteur de la grille
            sigma: Écart-type pour le blur gaussien
        """
        self.width = width or config.heatmap.grid_width
        self.height = height or config.heatmap.grid_height
        self.sigma = sigma or config.heatmap.blur_sigma

        # Grilles par joueur
        self.player_grids: Dict[int, np.ndarray] = {}

        # Grilles par équipe
        self.team_grids: Dict[str, np.ndarray] = {}

        # Grille globale
        self.global_grid = np.zeros((self.height, self.width), dtype=np.float32)

        # Facteurs d'échelle pour conversion pixels -> grille
        self.scale_x = 1.0
        self.scale_y = 1.0

    def set_video_dimensions(self, video_width: int, video_height: int):
        """
        Définir les dimensions de la vidéo pour la conversion

        Args:
            video_width: Largeur de la vidéo
            video_height: Hauteur de la vidéo
        """
        self.scale_x = self.width / video_width
        self.scale_y = self.height / video_height

    def add_position(
        self,
        player_id: int,
        position: Tuple[float, float],
        team: Optional[str] = None,
        weight: float = 1.0
    ):
        """
        Ajouter une position à la heatmap

        Args:
            player_id: ID du joueur
            position: Position (x, y) en pixels vidéo
            team: Équipe du joueur
            weight: Poids de la position
        """
        # Convertir en coordonnées grille
        x = int(position[0] * self.scale_x)
        y = int(position[1] * self.scale_y)

        # Vérifier les limites
        if 0 <= x < self.width and 0 <= y < self.height:
            # Grille du joueur
            if player_id not in self.player_grids:
                self.player_grids[player_id] = np.zeros(
                    (self.height, self.width), dtype=np.float32
                )
            self.player_grids[player_id][y, x] += weight

            # Grille de l'équipe
            if team:
                if team not in self.team_grids:
                    self.team_grids[team] = np.zeros(
                        (self.height, self.width), dtype=np.float32
                    )
                self.team_grids[team][y, x] += weight

            # Grille globale
            self.global_grid[y, x] += weight

    def generate_heatmap(
        self,
        player_id: Optional[int] = None,
        team: Optional[str] = None,
        normalize: bool = True
    ) -> HeatmapData:
        """
        Générer une heatmap

        Args:
            player_id: ID du joueur (None pour tous)
            team: Équipe (None pour toutes)
            normalize: Normaliser les valeurs entre 0 et 1

        Returns:
            HeatmapData
        """
        # Sélectionner la grille appropriée
        if player_id is not None:
            grid = self.player_grids.get(
                player_id,
                np.zeros((self.height, self.width), dtype=np.float32)
            ).copy()
        elif team is not None:
            grid = self.team_grids.get(
                team,
                np.zeros((self.height, self.width), dtype=np.float32)
            ).copy()
        else:
            grid = self.global_grid.copy()

        # Appliquer le blur gaussien
        if self.sigma > 0:
            grid = gaussian_filter(grid, sigma=self.sigma)

        # Normaliser
        if normalize and grid.max() > 0:
            grid = grid / grid.max()

        return HeatmapData(
            grid=grid,
            player_id=player_id,
            team=team,
            normalized=normalize
        )

    def render_heatmap(
        self,
        heatmap_data: HeatmapData,
        colormap: str = None,
        alpha: float = None,
        background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Rendre une heatmap en image

        Args:
            heatmap_data: Données de la heatmap
            colormap: Nom du colormap matplotlib
            alpha: Transparence
            background: Image de fond (optionnel)

        Returns:
            Image BGR de la heatmap
        """
        colormap = colormap or config.heatmap.colormap
        alpha = alpha or config.heatmap.alpha

        # Créer le colormap
        cmap = plt.get_cmap(colormap)

        # Appliquer le colormap
        colored = cmap(heatmap_data.grid)[:, :, :3]  # RGB, 0-1
        colored = (colored * 255).astype(np.uint8)
        colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

        # Redimensionner si nécessaire
        if background is not None:
            h, w = background.shape[:2]
            colored = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)

            # Créer un masque basé sur l'intensité
            mask = cv2.resize(
                heatmap_data.grid,
                (w, h),
                interpolation=cv2.INTER_LINEAR
            )
            mask = (mask * alpha).clip(0, 1)
            mask = np.stack([mask] * 3, axis=-1)

            # Fusionner avec le fond
            result = (
                background.astype(np.float32) * (1 - mask) +
                colored.astype(np.float32) * mask
            ).astype(np.uint8)
            return result

        return colored

    def render_on_pitch(
        self,
        heatmap_data: HeatmapData,
        pitch_image: np.ndarray,
        colormap: str = None,
        alpha: float = None
    ) -> np.ndarray:
        """
        Rendre une heatmap sur une image de terrain

        Args:
            heatmap_data: Données de la heatmap
            pitch_image: Image du terrain 2D
            colormap: Nom du colormap
            alpha: Transparence

        Returns:
            Image du terrain avec heatmap
        """
        return self.render_heatmap(heatmap_data, colormap, alpha, pitch_image)

    def get_hot_zones(
        self,
        heatmap_data: HeatmapData,
        threshold: float = 0.7,
        min_area: int = 10
    ) -> List[Tuple[int, int, int, int]]:
        """
        Obtenir les zones chaudes (régions de haute activité)

        Args:
            heatmap_data: Données de la heatmap
            threshold: Seuil de chaleur (0-1)
            min_area: Aire minimum des zones

        Returns:
            Liste de bounding boxes (x, y, w, h)
        """
        # Binariser
        binary = (heatmap_data.grid > threshold).astype(np.uint8) * 255

        # Trouver les contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        zones = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                zones.append((x, y, w, h))

        return zones

    def get_average_position(
        self,
        player_id: Optional[int] = None,
        team: Optional[str] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Calculer la position moyenne

        Args:
            player_id: ID du joueur
            team: Équipe

        Returns:
            Position moyenne (x, y) en coordonnées grille
        """
        if player_id is not None:
            grid = self.player_grids.get(player_id)
        elif team is not None:
            grid = self.team_grids.get(team)
        else:
            grid = self.global_grid

        if grid is None or grid.sum() == 0:
            return None

        # Centre de masse pondéré
        y_coords, x_coords = np.mgrid[0:self.height, 0:self.width]
        total = grid.sum()
        x_avg = (x_coords * grid).sum() / total
        y_avg = (y_coords * grid).sum() / total

        return (x_avg, y_avg)

    def get_coverage_percentage(
        self,
        player_id: Optional[int] = None,
        team: Optional[str] = None,
        threshold: float = 0.1
    ) -> float:
        """
        Calculer le pourcentage de couverture du terrain

        Args:
            player_id: ID du joueur
            team: Équipe
            threshold: Seuil minimum d'activité

        Returns:
            Pourcentage de couverture (0-100)
        """
        heatmap = self.generate_heatmap(player_id, team, normalize=True)
        covered = (heatmap.grid > threshold).sum()
        total = self.width * self.height
        return (covered / total) * 100

    def compare_teams(
        self,
        team_a: str,
        team_b: str
    ) -> np.ndarray:
        """
        Créer une heatmap comparative entre deux équipes

        Args:
            team_a: Nom de l'équipe A
            team_b: Nom de l'équipe B

        Returns:
            Grille de différence (-1 = B dominant, +1 = A dominant)
        """
        heatmap_a = self.generate_heatmap(team=team_a, normalize=True)
        heatmap_b = self.generate_heatmap(team=team_b, normalize=True)

        # Différence normalisée
        diff = heatmap_a.grid - heatmap_b.grid
        max_val = max(abs(diff.min()), abs(diff.max()))
        if max_val > 0:
            diff = diff / max_val

        return diff

    def render_comparison(
        self,
        team_a: str,
        team_b: str,
        background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Rendre une comparaison visuelle entre deux équipes

        Args:
            team_a: Nom de l'équipe A
            team_b: Nom de l'équipe B
            background: Image de fond

        Returns:
            Image de la comparaison
        """
        diff = self.compare_teams(team_a, team_b)

        # Colormap divergent (bleu = B, rouge = A)
        cmap = plt.get_cmap('RdBu_r')

        # Normaliser entre 0 et 1 pour le colormap
        normalized = (diff + 1) / 2

        colored = cmap(normalized)[:, :, :3]
        colored = (colored * 255).astype(np.uint8)
        colored = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

        if background is not None:
            h, w = background.shape[:2]
            colored = cv2.resize(colored, (w, h))

            # Masque basé sur l'activité totale
            total = self.generate_heatmap(normalize=True).grid
            mask = cv2.resize(total, (w, h))
            mask = (mask * 0.7).clip(0, 1)
            mask = np.stack([mask] * 3, axis=-1)

            result = (
                background.astype(np.float32) * (1 - mask) +
                colored.astype(np.float32) * mask
            ).astype(np.uint8)
            return result

        return colored

    def reset(self):
        """Réinitialiser toutes les données"""
        self.player_grids.clear()
        self.team_grids.clear()
        self.global_grid = np.zeros((self.height, self.width), dtype=np.float32)

    def reset_player(self, player_id: int):
        """Réinitialiser les données d'un joueur"""
        if player_id in self.player_grids:
            del self.player_grids[player_id]

    def save_heatmap(
        self,
        heatmap_data: HeatmapData,
        output_path: str,
        colormap: str = None
    ):
        """
        Sauvegarder une heatmap en image

        Args:
            heatmap_data: Données de la heatmap
            output_path: Chemin de sortie
            colormap: Nom du colormap
        """
        image = self.render_heatmap(heatmap_data, colormap)
        cv2.imwrite(output_path, image)

    def export_data(self, output_path: str):
        """
        Exporter les données brutes en fichier numpy

        Args:
            output_path: Chemin de sortie (.npz)
        """
        data = {
            'global': self.global_grid,
            'width': self.width,
            'height': self.height,
            'sigma': self.sigma
        }

        for player_id, grid in self.player_grids.items():
            data[f'player_{player_id}'] = grid

        for team, grid in self.team_grids.items():
            data[f'team_{team}'] = grid

        np.savez(output_path, **data)
