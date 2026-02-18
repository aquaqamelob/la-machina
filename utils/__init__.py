"""
Football Analysis - Utility modules
"""

from .colors import ColorPalette, get_team_color, interpolate_color
from .geometry import (
    calculate_distance,
    calculate_angle,
    point_in_polygon,
    line_intersection,
    smooth_trajectory
)
from .video_utils import (
    get_video_info,
    resize_frame,
    create_video_writer
)

__all__ = [
    'ColorPalette',
    'get_team_color',
    'interpolate_color',
    'calculate_distance',
    'calculate_angle',
    'point_in_polygon',
    'line_intersection',
    'smooth_trajectory',
    'get_video_info',
    'resize_frame',
    'create_video_writer'
]
