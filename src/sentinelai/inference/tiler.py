"""Spatial sliding-window tiling and coordinate generation module.

Provides deterministic sliding-window crop generation covering arbitrary image sizes,
supporting rectangular tiles, independent horizontal/vertical overlap ratios,
boundary-shift preservation, and deduplication.
"""

from typing import List, Optional, Tuple, Set
from sentinelai.dataset.structures import TileCrop


class SpatialTiler:
    """Generates overlapping sliding window crops over an image with deterministic boundary handling.

    Attributes:
        tile_width (int): Width of each tile in pixels.
        tile_height (int): Height of each tile in pixels.
        overlap_x (float): Horizontal overlap ratio in [0, 1).
        overlap_y (float): Vertical overlap ratio in [0, 1).
    """

    def __init__(
        self,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        tile_size: Optional[int] = None,
        overlap_x: Optional[float] = None,
        overlap_y: Optional[float] = None,
        overlap_ratio: Optional[float] = None,
    ):
        """Initializes the spatial sliding-window tiler.

        Args:
            tile_width: Width of tile in pixels. If None, uses tile_size.
            tile_height: Height of tile in pixels. If None, uses tile_size.
            tile_size: Square tile dimension fallback (default: 400).
            overlap_x: Horizontal overlap ratio in [0, 1). If None, uses overlap_ratio.
            overlap_y: Vertical overlap ratio in [0, 1). If None, uses overlap_ratio.
            overlap_ratio: Overlap ratio fallback in [0, 1) (default: 0.20).

        Raises:
            ValueError: If dimensions <= 0 or overlap ratios not in [0, 1).
        """
        # Resolve tile dimensions
        if tile_width is None and tile_height is None:
            resolved_size = 400 if tile_size is None else tile_size
            self.tile_width = int(resolved_size)
            self.tile_height = int(resolved_size)
        else:
            self.tile_width = int(tile_width if tile_width is not None else (tile_size or 400))
            self.tile_height = int(tile_height if tile_height is not None else (tile_size or 400))

        if self.tile_width <= 0:
            raise ValueError(f"tile_width must be positive, got {self.tile_width}")
        if self.tile_height <= 0:
            raise ValueError(f"tile_height must be positive, got {self.tile_height}")

        # Resolve overlap ratios
        resolved_overlap = 0.20 if overlap_ratio is None else float(overlap_ratio)
        self.overlap_x = float(overlap_x if overlap_x is not None else resolved_overlap)
        self.overlap_y = float(overlap_y if overlap_y is not None else resolved_overlap)

        if not (0.0 <= self.overlap_x < 1.0):
            raise ValueError(f"overlap_x must be in [0, 1), got {self.overlap_x}")
        if not (0.0 <= self.overlap_y < 1.0):
            raise ValueError(f"overlap_y must be in [0, 1), got {self.overlap_y}")

    @property
    def tile_size(self) -> int:
        """Square size shortcut for backwards compatibility."""
        return max(self.tile_width, self.tile_height)

    @property
    def overlap_ratio(self) -> float:
        """Mean overlap ratio for backwards compatibility."""
        return (self.overlap_x + self.overlap_y) / 2.0

    def generate_tiles(self, img_width: int, img_height: int) -> List[TileCrop]:
        """Generates deterministic tile bounding boxes covering [0, 0, img_width, img_height].

        Guarantees:
        1. All pixels in the image are covered by at least one tile.
        2. When img_dim >= tile_dim, edge tiles are shifted backwards so every tile
           retains the exact requested tile dimensions (preserving receptive-field uniformity).
        3. When img_dim < tile_dim, the tile is clipped to the available image boundary.
        4. No duplicate tiles are generated.

        Args:
            img_width: Image width in pixels.
            img_height: Image height in pixels.

        Returns:
            List of TileCrop instances ordered top-to-bottom, left-to-right.
        """
        if img_width <= 0 or img_height <= 0:
            return []

        # Case 1: Image is smaller than or equal to tile in both dimensions
        if img_width <= self.tile_width and img_height <= self.tile_height:
            return [TileCrop(tile_id=0, xmin=0, ymin=0, xmax=img_width, ymax=img_height)]

        # Calculate horizontal strides and coordinates
        stride_x = max(1, int(round(self.tile_width * (1.0 - self.overlap_x))))
        x_coords: List[int] = []
        if img_width <= self.tile_width:
            x_coords = [0]
        else:
            x = 0
            while x + self.tile_width <= img_width:
                x_coords.append(x)
                x += stride_x
            # Guarantee edge column coverage by shifting backwards from right border
            last_x = max(0, img_width - self.tile_width)
            if not x_coords or last_x > x_coords[-1]:
                x_coords.append(last_x)

        # Calculate vertical strides and coordinates
        stride_y = max(1, int(round(self.tile_height * (1.0 - self.overlap_y))))
        y_coords: List[int] = []
        if img_height <= self.tile_height:
            y_coords = [0]
        else:
            y = 0
            while y + self.tile_height <= img_height:
                y_coords.append(y)
                y += stride_y
            # Guarantee edge row coverage by shifting backwards from bottom border
            last_y = max(0, img_height - self.tile_height)
            if not y_coords or last_y > y_coords[-1]:
                y_coords.append(last_y)

        # Deduplicate coordinates while preserving order
        x_coords = sorted(list(set(x_coords)))
        y_coords = sorted(list(set(y_coords)))

        tiles: List[TileCrop] = []
        seen_coords: Set[Tuple[int, int, int, int]] = set()
        tile_id = 0

        for y0 in y_coords:
            y1 = min(img_height, y0 + self.tile_height)
            for x0 in x_coords:
                x1 = min(img_width, x0 + self.tile_width)
                box_tuple = (x0, y0, x1, y1)
                if box_tuple not in seen_coords:
                    seen_coords.add(box_tuple)
                    tiles.append(TileCrop(
                        tile_id=tile_id,
                        xmin=x0,
                        ymin=y0,
                        xmax=x1,
                        ymax=y1
                    ))
                    tile_id += 1

        return tiles
