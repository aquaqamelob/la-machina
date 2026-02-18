"""
Football Analysis - Dashboard Streamlit
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import json
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class FootballDashboard:
    """
    Dashboard Streamlit pour l'analyse de football
    """

    def __init__(self):
        """Initialiser le dashboard"""
        self.analyzer = None
        self.video_path = None
        self.output_path = None

    def run(self):
        """Lancer le dashboard"""
        st.set_page_config(
            page_title="Football Match Analysis",
            page_icon="⚽",
            layout="wide"
        )

        st.title("⚽ Football Match Analysis")
        st.markdown("Analyse de matchs de football avec YOLO et Computer Vision")

        # Sidebar pour les options
        with st.sidebar:
            st.header("Configuration")
            self._render_sidebar()

        # Contenu principal
        tab1, tab2, tab3, tab4 = st.tabs([
            "📹 Analyse Vidéo",
            "📊 Statistiques",
            "🔥 Heatmaps",
            "📈 Visualisations"
        ])

        with tab1:
            self._render_video_tab()

        with tab2:
            self._render_stats_tab()

        with tab3:
            self._render_heatmap_tab()

        with tab4:
            self._render_viz_tab()

    def _render_sidebar(self):
        """Rendre la sidebar de configuration"""
        st.subheader("Source Vidéo")

        source_type = st.radio(
            "Type de source",
            ["Upload", "URL YouTube", "Fichier local"]
        )

        if source_type == "Upload":
            uploaded_file = st.file_uploader(
                "Choisir une vidéo",
                type=["mp4", "avi", "mov", "mkv"]
            )
            if uploaded_file:
                # Sauvegarder temporairement
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    f.write(uploaded_file.read())
                    self.video_path = f.name
                st.success("Vidéo chargée!")

        elif source_type == "URL YouTube":
            youtube_url = st.text_input("URL YouTube")
            if youtube_url and st.button("Télécharger"):
                with st.spinner("Téléchargement en cours..."):
                    from utils.video_utils import download_youtube_video
                    self.video_path = download_youtube_video(
                        youtube_url,
                        tempfile.gettempdir(),
                        max_duration=300  # 5 minutes max
                    )
                    if self.video_path:
                        st.success("Vidéo téléchargée!")
                    else:
                        st.error("Échec du téléchargement")

        else:
            local_path = st.text_input("Chemin du fichier")
            if local_path and Path(local_path).exists():
                self.video_path = local_path
                st.success("Fichier trouvé!")

        st.divider()

        st.subheader("Options d'analyse")

        self.enable_pose = st.checkbox("Estimation de pose", value=False)
        self.enable_speed = st.checkbox("Calcul de vitesse", value=True)
        self.enable_heatmap = st.checkbox("Génération heatmaps", value=True)
        self.enable_stats = st.checkbox("Statistiques", value=True)

        st.divider()

        self.max_frames = st.number_input(
            "Max frames (0 = tous)",
            min_value=0,
            max_value=10000,
            value=0
        )

        self.skip_frames = st.number_input(
            "Skip frames",
            min_value=0,
            max_value=10,
            value=0
        )

    def _render_video_tab(self):
        """Rendre l'onglet d'analyse vidéo"""
        st.header("Analyse Vidéo")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vidéo source")
            if self.video_path:
                st.video(self.video_path)
            else:
                st.info("Chargez une vidéo depuis la sidebar")

        with col2:
            st.subheader("Vidéo analysée")
            if self.output_path and Path(self.output_path).exists():
                st.video(self.output_path)
            else:
                st.info("Lancez l'analyse pour voir le résultat")

        st.divider()

        if self.video_path:
            if st.button("🚀 Lancer l'analyse", type="primary"):
                self._run_analysis()

    def _run_analysis(self):
        """Lancer l'analyse de la vidéo"""
        if not self.video_path:
            st.error("Aucune vidéo chargée")
            return

        # Créer le fichier de sortie
        output_dir = Path(tempfile.gettempdir()) / "football_analysis"
        output_dir.mkdir(exist_ok=True)
        self.output_path = str(output_dir / "analyzed.mp4")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            from main import FootballAnalyzer

            # Initialiser l'analyseur
            analyzer = FootballAnalyzer(
                enable_pose=self.enable_pose,
                enable_speed=self.enable_speed,
                enable_heatmap=self.enable_heatmap,
                enable_minimap=True,
                enable_stats=self.enable_stats
            )

            # Lancer l'analyse
            status_text.text("Initialisation...")

            max_frames = self.max_frames if self.max_frames > 0 else None

            # Note: Pour un vrai streaming, il faudrait modifier
            # la méthode analyze_video pour supporter les callbacks
            analyzer.analyze_video(
                self.video_path,
                self.output_path,
                max_frames=max_frames,
                skip_frames=self.skip_frames
            )

            progress_bar.progress(100)
            status_text.text("Analyse terminée!")
            st.success("Analyse terminée avec succès!")

            # Sauvegarder l'analyseur pour les stats
            self.analyzer = analyzer

            st.rerun()

        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {e}")

    def _render_stats_tab(self):
        """Rendre l'onglet statistiques"""
        st.header("Statistiques du Match")

        # Charger les stats si disponibles
        output_dir = Path(tempfile.gettempdir()) / "football_analysis"
        stats_path = output_dir / "match_stats.json"

        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

            # Métriques principales
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Durée analysée",
                    f"{stats['duration']:.1f}s"
                )

            with col2:
                st.metric(
                    "Frames traités",
                    stats['frames_analyzed']
                )

            with col3:
                st.metric(
                    "Distance ballon",
                    f"{stats['ball_distance']:.0f}m"
                )

            with col4:
                st.metric(
                    "Changements de possession",
                    stats['possession_changes']
                )

            st.divider()

            # Stats par équipe
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔵 Équipe A")
                team_a = stats['team_a']
                st.metric("Possession", f"{team_a['possession']:.1f}%")
                st.metric("Distance totale", f"{team_a['total_distance']:.0f}m")
                st.metric("Sprints", team_a['sprints'])
                st.metric("Vitesse max", f"{team_a['max_speed']:.1f} km/h")

            with col2:
                st.subheader("🔴 Équipe B")
                team_b = stats['team_b']
                st.metric("Possession", f"{team_b['possession']:.1f}%")
                st.metric("Distance totale", f"{team_b['total_distance']:.0f}m")
                st.metric("Sprints", team_b['sprints'])
                st.metric("Vitesse max", f"{team_b['max_speed']:.1f} km/h")

            st.divider()

            # Tableau des joueurs
            st.subheader("Statistiques par joueur")
            players_df = pd.DataFrame([
                {
                    "ID": pid,
                    "Équipe": data["team"],
                    "Distance (m)": round(data["distance"], 1),
                    "Vitesse max (km/h)": round(data["max_speed"], 1),
                    "Vitesse moy (km/h)": round(data["avg_speed"], 1),
                    "Sprints": data["sprints"]
                }
                for pid, data in stats['players'].items()
            ])

            st.dataframe(players_df, use_container_width=True)

        else:
            st.info("Lancez une analyse pour voir les statistiques")

    def _render_heatmap_tab(self):
        """Rendre l'onglet heatmaps"""
        st.header("Heatmaps")

        output_dir = Path(tempfile.gettempdir()) / "football_analysis"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Heatmap globale")
            heatmap_path = output_dir / "heatmap_global.png"
            if heatmap_path.exists():
                st.image(str(heatmap_path))
            else:
                st.info("Non disponible")

        with col2:
            st.subheader("🔵 Équipe A")
            heatmap_path = output_dir / "heatmap_team_a.png"
            if heatmap_path.exists():
                st.image(str(heatmap_path))
            else:
                st.info("Non disponible")

        with col3:
            st.subheader("🔴 Équipe B")
            heatmap_path = output_dir / "heatmap_team_b.png"
            if heatmap_path.exists():
                st.image(str(heatmap_path))
            else:
                st.info("Non disponible")

    def _render_viz_tab(self):
        """Rendre l'onglet visualisations"""
        st.header("Visualisations")

        output_dir = Path(tempfile.gettempdir()) / "football_analysis"
        stats_path = output_dir / "match_stats.json"

        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

            # Graphique de possession
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Possession")
                fig = go.Figure(data=[go.Pie(
                    labels=['Équipe A', 'Équipe B'],
                    values=[stats['team_a']['possession'], stats['team_b']['possession']],
                    marker_colors=['#3498db', '#e74c3c'],
                    hole=0.4
                )])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Distance par équipe")
                fig = go.Figure(data=[go.Bar(
                    x=['Équipe A', 'Équipe B'],
                    y=[stats['team_a']['total_distance'], stats['team_b']['total_distance']],
                    marker_color=['#3498db', '#e74c3c']
                )])
                fig.update_layout(height=300, yaxis_title="Distance (m)")
                st.plotly_chart(fig, use_container_width=True)

            # Graphique des sprints
            st.subheader("Comparaison des équipes")
            categories = ['Possession (%)', 'Distance (x10m)', 'Sprints', 'Vitesse max (km/h)']

            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[
                    stats['team_a']['possession'],
                    stats['team_a']['total_distance'] / 10,
                    stats['team_a']['sprints'],
                    stats['team_a']['max_speed']
                ],
                theta=categories,
                fill='toself',
                name='Équipe A',
                line_color='#3498db'
            ))
            fig.add_trace(go.Scatterpolar(
                r=[
                    stats['team_b']['possession'],
                    stats['team_b']['total_distance'] / 10,
                    stats['team_b']['sprints'],
                    stats['team_b']['max_speed']
                ],
                theta=categories,
                fill='toself',
                name='Équipe B',
                line_color='#e74c3c'
            ))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True)))
            st.plotly_chart(fig, use_container_width=True)

            # Distribution des vitesses
            if stats['players']:
                st.subheader("Distribution des vitesses max par joueur")

                speeds_a = [
                    data['max_speed']
                    for data in stats['players'].values()
                    if data['team'] == 'team_a'
                ]
                speeds_b = [
                    data['max_speed']
                    for data in stats['players'].values()
                    if data['team'] == 'team_b'
                ]

                fig = go.Figure()
                if speeds_a:
                    fig.add_trace(go.Box(y=speeds_a, name='Équipe A', marker_color='#3498db'))
                if speeds_b:
                    fig.add_trace(go.Box(y=speeds_b, name='Équipe B', marker_color='#e74c3c'))
                fig.update_layout(yaxis_title="Vitesse max (km/h)")
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("Lancez une analyse pour voir les visualisations")


def main():
    """Point d'entrée du dashboard"""
    dashboard = FootballDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
