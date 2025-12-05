"""
DNO Library - библиотека для обработки и дополнения поз скелета
"""

from .core import (
    normalize_keypoints,
    PoseProcessor,
)

__version__ = "0.0.0"
__all__ = [
    'normalize_keypoints',
    'PoseProcessor',
]