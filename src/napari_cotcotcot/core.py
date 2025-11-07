#  Import required libraries

from typing import Optional

import napari
import numpy as np
import tifffile
import torch
from tqdm import tqdm

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CoTracker utils using device: {device}")

# https://github.com/facebookresearch/co-tracker/blob/main/notebooks/demo.ipynb
# Also load cotracker model here if not already done
cotracker = torch.hub.load(
    "facebookresearch/co-tracker", "cotracker3_offline"
).to(device)


def load_timelapse(filepath):
    """Load a time-lapse TIF file"""
    with tifffile.TiffFile(filepath) as tif:
        images = tif.asarray()

    if images.ndim == 3:
        print(f"Loaded grayscale time-lapse: {images.shape} (T, H, W)")
    elif images.ndim == 4:
        print(f"Loaded color time-lapse: {images.shape} (T, H, W, C)")
    else:
        raise ValueError(f"Unexpected image dimensions: {images.ndim}")

    return images


def get_image_from_layer(
    viewer: napari.Viewer, layer_name: str = None
) -> np.ndarray:
    """
    Get image data from a napari layer.

    Args:
        viewer: Napari viewer instance
        layer_name: Name of the image layer. If None, uses the first image layer.

    Returns:
        Image array with shape (T, H, W) or (T, H, W, C)
    """
    if layer_name is not None:
        # Get specific layer by name
        if layer_name not in viewer.layers:
            raise ValueError(f"Layer '{layer_name}' not found in viewer")
        layer = viewer.layers[layer_name]

        if not isinstance(layer, napari.layers.Image):
            raise ValueError(f"Layer '{layer_name}' is not an Image layer")
    else:
        # Find the first image layer
        image_layers = [
            layer
            for layer in viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]
        if not image_layers:
            raise ValueError("No image layers found in viewer")
        layer = image_layers[0]
        layer_name = layer.name

    # Get the image data
    images = layer.data

    # Ensure we have the right dimensions
    if images.ndim == 2:
        # Single 2D image - add time dimension
        images = images[np.newaxis, ...]
        print(
            f"Got single frame from layer '{layer_name}': {images.shape} (T=1, H, W)"
        )
    elif images.ndim == 3:
        # Could be (T, H, W) or (H, W, C)
        # Check if last dimension could be color channels
        if images.shape[-1] in [3, 4]:
            # Likely (H, W, C) - single frame with color
            images = images[np.newaxis, ...]
            print(
                f"Got single color frame from layer '{layer_name}': {images.shape} (T=1, H, W, C)"
            )
        else:
            # Likely (T, H, W) - time series
            print(
                f"Got grayscale time-lapse from layer '{layer_name}': {images.shape} (T, H, W)"
            )
    elif images.ndim == 4:
        # (T, H, W, C)
        print(
            f"Got color time-lapse from layer '{layer_name}': {images.shape} (T, H, W, C)"
        )
    else:
        raise ValueError(
            f"Unexpected image dimensions from layer '{layer_name}': {images.ndim}"
        )

    return images


def prepare_images_for_tracking(images: np.ndarray) -> np.ndarray:
    """
    Prepare images for tracking by converting to RGB if necessary.

    Args:
        images: Image array with shape (T, H, W) or (T, H, W, C)

    Returns:
        RGB image array with shape (T, H, W, 3)
    """
    if images.ndim == 3:
        # Convert grayscale to RGB
        images_rgb = np.stack([images, images, images], axis=-1)
        print(f"Converted grayscale to RGB: {images_rgb.shape}")
    elif images.ndim == 4:
        if images.shape[-1] == 3:
            # Already RGB
            images_rgb = images
            print(f"Images already RGB: {images_rgb.shape}")
        elif images.shape[-1] == 4:
            # RGBA - drop alpha channel
            images_rgb = images[..., :3]
            print(f"Converted RGBA to RGB: {images_rgb.shape}")
        elif images.shape[-1] == 1:
            # Single channel as (T, H, W, 1)
            images_rgb = np.repeat(images, 3, axis=-1)
            print(f"Converted single channel to RGB: {images_rgb.shape}")
        else:
            raise ValueError(
                f"Unexpected number of channels: {images.shape[-1]}"
            )
    else:
        raise ValueError(f"Unexpected image dimensions: {images.ndim}")

    return images_rgb


def generate_grid_points_in_rectangle(
    rect_coords: np.ndarray, grid_size: int = 5
) -> np.ndarray:
    """Generate a grid of points within a rectangle."""
    y_coords = rect_coords[:, 0]
    x_coords = rect_coords[:, 1]

    y_min, y_max = y_coords.min(), y_coords.max()
    x_min, x_max = x_coords.min(), x_coords.max()

    y_points = np.linspace(y_min, y_max, grid_size)
    x_points = np.linspace(x_min, x_max, grid_size)

    xx, yy = np.meshgrid(x_points, y_points)
    points = np.column_stack([yy.flatten(), xx.flatten()])

    return points


def generate_grid_points_in_disk(
    center_y: float, center_x: float, radius: float, grid_size: int = 5
) -> np.ndarray:
    """Generate a grid of points within a disk."""
    # Create a square grid
    points_1d = np.linspace(-radius, radius, grid_size)
    xx, yy = np.meshgrid(points_1d, points_1d)

    # Flatten and filter points within the circle
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()

    # Keep only points within the circle
    distances = np.sqrt(xx_flat**2 + yy_flat**2)
    mask = distances <= radius

    # Create points array
    points = np.column_stack(
        [yy_flat[mask] + center_y, xx_flat[mask] + center_x]
    )

    return points


def generate_rectangle_from_center(
    center_y: float, center_x: float, width: float = 50, height: float = 50
) -> np.ndarray:
    """Generate rectangle coordinates from a center point."""
    half_w = width / 2
    half_h = height / 2

    return np.array(
        [
            [center_y - half_h, center_x - half_w],
            [center_y - half_h, center_x + half_w],
            [center_y + half_h, center_x + half_w],
            [center_y + half_h, center_x - half_w],
        ]
    )


def generate_disk_polygon(
    center_y: float, center_x: float, radius: float, n_points: int = 32
) -> np.ndarray:
    """Generate disk boundary as polygon points."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    points = np.column_stack(
        [
            center_y + radius * np.sin(angles),
            center_x + radius * np.cos(angles),
        ]
    )
    return points


def prepare_video_for_cotracker(images_rgb: np.ndarray) -> torch.Tensor:
    """Prepare video tensor for CoTracker."""
    if images_rgb.dtype != np.uint8:
        if images_rgb.max() <= 1.0:
            images_rgb = (images_rgb * 255).astype(np.uint8)
        else:
            images_rgb = np.clip(images_rgb, 0, 255).astype(np.uint8)

    video = (
        torch.tensor(images_rgb).permute(0, 3, 1, 2)[None].float().to(device)
    )
    return video


# Tracking functions
def track_single_seed_layer(
    images_rgb: np.ndarray,
    seed_points: np.ndarray,
    shape_type: str = "rectangle",
    shape_size: float = 50,
    grid_size: int = 5,
    layer_name: str = "",
) -> list[dict]:
    """Track a single seed layer between its seed points."""
    if len(seed_points) == 0:
        return []

    video = prepare_video_for_cotracker(images_rgb)

    # Sort seed points by time
    sorted_indices = np.argsort(seed_points[:, 0])
    sorted_seeds = seed_points[sorted_indices]

    results = []

    # Progress bar for tracking segments
    pbar = tqdm(range(len(sorted_seeds) - 1), desc=f"Tracking {layer_name}")

    for i in pbar:
        seed_point = sorted_seeds[i]
        t_start, y_start, x_start = seed_point
        t_start = int(t_start)

        # Track to next seed point
        t_end = int(sorted_seeds[i + 1, 0])

        if t_end <= t_start:
            continue

        pbar.set_postfix({"frames": f"{t_start}-{t_end}"})

        # Generate shape and grid points
        if shape_type == "rectangle":
            shape_coords = generate_rectangle_from_center(
                y_start, x_start, shape_size, shape_size
            )
            grid_points = generate_grid_points_in_rectangle(
                shape_coords, grid_size
            )
        else:  # disk
            shape_coords = generate_disk_polygon(
                y_start, x_start, shape_size / 2
            )
            grid_points = generate_grid_points_in_disk(
                y_start, x_start, shape_size / 2, grid_size
            )

        # Prepare queries
        N = len(grid_points)
        queries = torch.zeros((1, N, 3), device=device)
        queries[0, :, 0] = 0
        queries[0, :, 1] = torch.tensor(grid_points[:, 1], device=device)
        queries[0, :, 2] = torch.tensor(grid_points[:, 0], device=device)

        # Extract video segment and track
        video_segment = video[:, t_start:t_end]
        pred_tracks, pred_visibility = cotracker(
            video_segment, queries=queries
        )

        results.append(
            {
                "segment_idx": i,
                "seed_point": seed_point,
                "start_frame": t_start,
                "end_frame": t_end,
                "tracks": pred_tracks,
                "visibility": pred_visibility,
                "initial_points": grid_points,
                "shape_coords": shape_coords,
                "shape_type": shape_type,
                "shape_size": shape_size,
            }
        )

    return results


def compute_center_of_mass(
    tracks: np.ndarray, visibility: np.ndarray
) -> np.ndarray:
    """Compute center of mass for visible points at each frame."""
    T, N, _ = tracks.shape
    centers = []

    for t in range(T):
        visible_mask = visibility[t] > 0.5
        if visible_mask.sum() > 0:
            visible_points = tracks[t][visible_mask]
            center = visible_points.mean(axis=0)
            centers.append(center)
        else:
            centers.append([np.nan, np.nan])

    return np.array(centers)


# Visualization functions
def visualize_seed_layer_results(
    viewer,
    results: list[dict],
    seed_name: str,
    color: str,
    grid_size: int,
    show_shapes: bool,
    show_centers: bool,
):
    """Visualize tracking results for a single seed layer."""

    # Generate result layer names based on seed name
    base_name = seed_name.replace(
        "seed_", ""
    )  # Remove seed_ prefix if present

    # Clear existing layers for this seed
    layers_to_clear = [
        f"track_{base_name}",
        f"shape_{base_name}",
        f"center_{base_name}",
    ]

    for layer in layers_to_clear:
        if layer in viewer.layers:
            viewer.layers.remove(layer)

    if not results:
        return

    # Collect all data
    all_tracks_data = []
    all_shapes_data = []
    all_centers_data = []

    track_id_offset = 0

    for result in enumerate(results):
        tracks = result["tracks"][0].cpu().numpy()
        visibility = result["visibility"][0].cpu().numpy()
        shape_type = result["shape_type"]

        if visibility.ndim == 3:
            visibility = visibility[:, :, 0]

        T_segment, N, _ = tracks.shape
        start_frame = result["start_frame"]

        # Convert x,y to y,x for napari
        tracks = tracks[:, :, [1, 0]]

        # Add tracks
        for point_idx in range(N):
            for t_local in range(T_segment):
                if visibility[t_local, point_idx] > 0.5:
                    t_global = start_frame + t_local
                    all_tracks_data.append(
                        [
                            track_id_offset + point_idx,
                            t_global,
                            tracks[t_local, point_idx, 0],
                            tracks[t_local, point_idx, 1],
                        ]
                    )

        # Compute center of mass
        if show_centers:
            centers = compute_center_of_mass(tracks, visibility)
            for t_local, center in enumerate(centers):
                if not np.isnan(center[0]):
                    t_global = start_frame + t_local
                    all_centers_data.append([t_global, center[0], center[1]])

        # Create deforming shapes for each frame
        if show_shapes:
            if shape_type == "rectangle":
                # Get corner indices
                points_per_rect = grid_size * grid_size
                corner_indices = [
                    0,  # top-left
                    grid_size - 1,  # top-right
                    points_per_rect - grid_size,  # bottom-left
                    points_per_rect - 1,  # bottom-right
                ]

                for t in range(T_segment):
                    corners_visible = all(
                        visibility[t, idx] > 0.5 for idx in corner_indices
                    )

                    if corners_visible:
                        corners = [tracks[t, idx] for idx in corner_indices]
                        corners_3d = np.column_stack(
                            [
                                np.full(4, start_frame + t),
                                [c[0] for c in corners],
                                [c[1] for c in corners],
                            ]
                        )
                        all_shapes_data.append(corners_3d)

            else:  # disk
                # Use convex hull of visible points
                for t in range(T_segment):
                    visible_mask = visibility[t] > 0.5
                    if (
                        visible_mask.sum() >= 8
                    ):  # Need enough points for a reasonable shape
                        visible_points = tracks[t][visible_mask]

                        # Simple approach: use points on the boundary
                        center = visible_points.mean(axis=0)
                        vectors = visible_points - center
                        distances = np.linalg.norm(vectors, axis=1)
                        max_dist = distances.max()

                        # Create disk polygon
                        n_points = 32
                        angles = np.linspace(
                            0, 2 * np.pi, n_points, endpoint=False
                        )
                        disk_points = np.column_stack(
                            [
                                center[0] + max_dist * np.sin(angles),
                                center[1] + max_dist * np.cos(angles),
                            ]
                        )

                        disk_3d = np.column_stack(
                            [
                                np.full(n_points, start_frame + t),
                                disk_points[:, 0],
                                disk_points[:, 1],
                            ]
                        )
                        all_shapes_data.append(disk_3d)

        track_id_offset += N

    # Add layers with consistent naming
    if all_tracks_data:
        all_tracks_data = np.array(all_tracks_data)
        viewer.add_tracks(
            all_tracks_data,
            name=f"track_{base_name}",
            tail_length=10,
            tail_width=2,
            head_length=0,
            colormap="turbo",
        )

    if show_shapes and all_shapes_data:
        viewer.add_shapes(
            all_shapes_data,
            shape_type="polygon",
            edge_color=color,
            edge_width=2,
            face_color=[0, 0, 0, 0],
            name=f"shape_{base_name}",
        )

    if show_centers and all_centers_data:
        all_centers_data = np.array(all_centers_data)
        viewer.add_points(
            all_centers_data,
            name=f"center_{base_name}",
            size=8,
            face_color=color,
            symbol="disc",
        )


# Improved Seed layer management
class SeedLayerManager:
    """Manage multiple seed point layers."""

    def __init__(self, viewer, seed_prefix: str = "seed_"):
        self.viewer = viewer
        self.seed_prefix = seed_prefix
        self.seed_layers = {}
        self.layer_colors = [
            "cyan",
            "magenta",
            "yellow",
            "lime",
            "orange",
            "purple",
            "pink",
            "green",
            "blue",
            "red",
        ]
        self.color_idx = 0

        # Scan for existing seed layers
        self.scan_existing_layers()

    def scan_existing_layers(self):
        """Scan viewer for existing seed layers and register them."""
        found_layers = []

        for layer in self.viewer.layers:
            if isinstance(
                layer, napari.layers.Points
            ) and layer.name.startswith(self.seed_prefix):

                # Assign color
                color = self.layer_colors[
                    self.color_idx % len(self.layer_colors)
                ]
                self.color_idx += 1

                # Register layer
                self.seed_layers[layer.name] = {
                    "layer": layer,
                    "color": color,
                    "results": None,
                }
                found_layers.append(layer.name)

        if found_layers:
            print(
                f"Found {len(found_layers)} existing seed layers: {found_layers}"
            )

    def register_existing_layer(self, layer_name: str, color: str = None):
        """Register an existing points layer as a seed layer."""
        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]

            if isinstance(layer, napari.layers.Points):
                if color is None:
                    color = self.layer_colors[
                        self.color_idx % len(self.layer_colors)
                    ]
                    self.color_idx += 1

                self.seed_layers[layer_name] = {
                    "layer": layer,
                    "color": color,
                    "results": None,
                }
                print(
                    f"Registered existing layer '{layer_name}' with color {color}"
                )
            else:
                print(f"Layer '{layer_name}' is not a points layer!")
        else:
            print(f"Layer '{layer_name}' not found in viewer!")

    def add_seed_layer(
        self, name: str = None, color: str = None
    ) -> napari.layers.Points:
        """Add a new seed point layer."""
        if name is None:
            # Generate unique name
            idx = len(self.seed_layers) + 1
            name = f"{self.seed_prefix}{idx}"
            while name in self.viewer.layers:
                idx += 1
                name = f"{self.seed_prefix}{idx}"
        elif not name.startswith(self.seed_prefix):
            name = f"{self.seed_prefix}{name}"

        if color is None:
            color = self.layer_colors[self.color_idx % len(self.layer_colors)]
            self.color_idx += 1

        layer = self.viewer.add_points(
            ndim=3, size=10, face_color=color, name=name
        )

        self.seed_layers[name] = {
            "layer": layer,
            "color": color,
            "results": None,
        }

        print(f"Added new seed layer '{name}' with color {color}")
        return layer

    def get_seed_points(self, layer_name: str) -> Optional[np.ndarray]:
        """Get seed points from a specific layer."""
        if layer_name in self.seed_layers:
            return self.seed_layers[layer_name]["layer"].data
        return None

    def get_all_seed_layers(self) -> list[str]:
        """Get names of all seed layers."""
        return list(self.seed_layers.keys())

    def refresh_layers(self):
        """Rescan for new seed layers."""
        self.scan_existing_layers()


#  Main tracking functions
# Modify track_seed_layer to accept images_rgb
def track_seed_layer(
    images_rgb: np.ndarray,
    seed_manager: SeedLayerManager,
    layer_name: str,
    shape_type: str = "rectangle",
    shape_size: float = 50,
    grid_size: int = 5,
    show_shapes: bool = True,
    show_centers: bool = True,
):
    """Track a specific seed layer."""

    seed_points = seed_manager.get_seed_points(layer_name)

    if seed_points is None:
        print(f"Seed layer '{layer_name}' not found!")
        print(f"Available layers: {seed_manager.get_all_seed_layers()}")
        return

    if len(seed_points) == 0:
        print(f"No seed points in layer '{layer_name}'!")
        return

    print(
        f"\nTracking layer '{layer_name}' with {len(seed_points)} seed points..."
    )

    # Run tracking - now images_rgb is passed as parameter
    results = track_single_seed_layer(
        images_rgb,
        seed_points,
        shape_type=shape_type,
        shape_size=shape_size,
        grid_size=grid_size,
        layer_name=layer_name,
    )

    # Store results
    seed_manager.seed_layers[layer_name]["results"] = results

    # Get viewer from seed_manager
    viewer = seed_manager.viewer

    # Visualize
    color = seed_manager.seed_layers[layer_name]["color"]
    visualize_seed_layer_results(
        viewer,
        results,
        layer_name,
        color,
        grid_size,
        show_shapes,
        show_centers,
    )

    print(f"Tracking complete for layer '{layer_name}'!")


# Also modify track_all_seed_layers
def track_all_seed_layers(
    images_rgb: np.ndarray,
    seed_manager: SeedLayerManager,
    shape_type: str = "rectangle",
    shape_size: float = 50,
    grid_size: int = 5,
    show_shapes: bool = True,
    show_centers: bool = True,
):
    """Track all seed layers."""

    layer_names = seed_manager.get_all_seed_layers()

    if not layer_names:
        print("No seed layers found! Add seed layers first.")
        return

    print(f"Tracking {len(layer_names)} seed layers...")

    for layer_name in layer_names:
        track_seed_layer(
            images_rgb,
            seed_manager,
            layer_name,
            shape_type=shape_type,
            shape_size=shape_size,
            grid_size=grid_size,
            show_shapes=show_shapes,
            show_centers=show_centers,
        )

    print("\nAll tracking complete!")


#  Export functions
def export_all_centers_to_csv(
    seed_manager: SeedLayerManager, filename: str = "all_centers.csv"
):
    """Export all center of mass positions to CSV file."""
    import pandas as pd

    data = []

    for layer_name, layer_info in seed_manager.seed_layers.items():
        results = layer_info.get("results", [])

        for result in results:
            tracks = result["tracks"][0].cpu().numpy()[:, :, [1, 0]]
            visibility = result["visibility"][0].cpu().numpy()
            if visibility.ndim == 3:
                visibility = visibility[:, :, 0]

            centers = compute_center_of_mass(tracks, visibility)
            start_frame = result["start_frame"]

            for t_local, center in enumerate(centers):
                if not np.isnan(center[0]):
                    data.append(
                        {
                            "layer": layer_name,
                            "frame": start_frame + t_local,
                            "y": center[0],
                            "x": center[1],
                        }
                    )

    if data:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Exported {len(data)} center positions to {filename}")
    else:
        print("No tracking results to export!")
