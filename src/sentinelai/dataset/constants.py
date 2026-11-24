"""DIOR dataset constants, class mappings, and size protocol definitions."""

from typing import Dict, List, Tuple

DIOR_CLASSES: List[str] = [
    "Expressway-Service-area",
    "Expressway-toll-station",
    "airplane",
    "airport",
    "baseballfield",
    "basketballcourt",
    "bridge",
    "chimney",
    "dam",
    "golffield",
    "groundtrackfield",
    "harbor",
    "overpass",
    "ship",
    "stadium",
    "storagetank",
    "tenniscourt",
    "trainstation",
    "vehicle",
    "windmill"
]

CLASS_NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(DIOR_CLASSES)}
CLASS_ID_TO_NAME: Dict[int, str] = {i: name for i, name in enumerate(DIOR_CLASSES)}

# Typical DIOR native image dimensions
DEFAULT_IMAGE_WIDTH: int = 800
DEFAULT_IMAGE_HEIGHT: int = 800

# Explicit Object Size Protocol
# Grounded in MS COCO convention adapted for 800x800 remote sensing imagery:
# - Small: area <= 32^2 px (1024 px^2), or normalized area <= 1024 / (800*800) = 0.0016
# - Medium: 32^2 < area <= 96^2 px (1024 to 9216 px^2), or normalized 0.0016 to 0.0144
# - Large: area > 96^2 px (> 9216 px^2), or normalized > 0.0144
SIZE_BINS: Dict[str, Tuple[float, float]] = {
    "small": (0.0, 32.0 ** 2),
    "medium": (32.0 ** 2, 96.0 ** 2),
    "large": (96.0 ** 2, float("inf")),
}

# Small-object representative classes in DIOR
TYPICAL_SMALL_CLASSES: List[str] = ["vehicle", "ship", "storagetank", "airplane", "windmill"]
