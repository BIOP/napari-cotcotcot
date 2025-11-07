"""
CoTracker widget for napari

This module provides a comprehensive GUI for using CoTracker
to track objects in time-lapse images using seed points.
"""

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Label,
    LineEdit,
    PushButton,
    SpinBox,
)
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import (
    QColorDialog,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    import napari

# Import your core functionality from the core module
from .core import (
    SeedLayerManager,
    get_image_from_layer,
    prepare_images_for_tracking,
    track_all_seed_layers,
    track_seed_layer,
)


class CoTrackerWidget(QWidget):
    """Main widget for CoTracker functionality in napari."""

    def __init__(
        self, viewer: "napari.viewer.Viewer"
    ):  # STRING ANNOTATION HERE!
        super().__init__()
        self.viewer = viewer
        self.seed_manager = None
        self.images_rgb = None
        self.layer_event_connections = {}  # Store event connections
        self.default_colors = [
            "cyan",
            "pink",
            "yellow",
            "lime",
            "orange",
            "purple",
            "green",
            "blue",
            "red",
        ]
        self.color_index = 0
        self.current_color = "cyan"  # Start with cyan

        # Create main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = Label(value="<h2>Cot Cot Cot</h2>")
        layout.addWidget(title.native)

        # Image selection section
        layout.addWidget(Label(value="<b>1. Select Image Layer:</b>").native)
        self.image_selector = ComboBox(
            label="Image Layer", choices=self._get_image_layers()
        )
        self.load_image_btn = PushButton(text="Load Selected Image")
        self.load_image_btn.clicked.connect(self._load_image)

        image_layout = QHBoxLayout()
        image_layout.addWidget(self.image_selector.native)
        image_layout.addWidget(self.load_image_btn.native)
        layout.addLayout(image_layout)

        self.image_info = Label(value="No image loaded")
        layout.addWidget(self.image_info.native)

        # Separator
        layout.addWidget(Label(value="<hr>").native)

        # Seed layer management section
        layout.addWidget(
            Label(value="<b>2. Seed Layer Management:</b>").native
        )

        # Add new seed layer with color selection
        seed_layout = QHBoxLayout()
        self.seed_name_input = LineEdit(value="SP1", label="Name")

        # Color selection - ComboBox with predefined colors
        self.color_selector = ComboBox(
            label="Color", choices=self.default_colors, value="cyan"
        )

        # Color picker button for custom colors
        self.color_picker_btn = QPushButton("Custom Color")
        self.color_picker_btn.clicked.connect(self._pick_custom_color)

        self.add_seed_btn = PushButton(text="Add New Seed Layer")
        self.add_seed_btn.clicked.connect(self._add_seed_layer)

        seed_layout.addWidget(Label(value="Name:").native)
        seed_layout.addWidget(self.seed_name_input.native)
        seed_layout.addWidget(Label(value="Color:").native)
        seed_layout.addWidget(self.color_selector.native)
        seed_layout.addWidget(self.color_picker_btn)
        seed_layout.addWidget(self.add_seed_btn.native)
        layout.addLayout(seed_layout)

        # Load seed from CSV button
        self.load_seed_btn = PushButton(text="Load Seed Layer from CSV")
        self.load_seed_btn.clicked.connect(self._load_seed_from_csv)
        layout.addWidget(self.load_seed_btn.native)

        # Time navigation section
        layout.addWidget(Label(value="<b>Time Navigation:</b>").native)
        nav_layout = QHBoxLayout()

        self.prev_frame_btn = PushButton(text="← Previous")
        self.prev_frame_btn.clicked.connect(self._previous_frame)
        self.next_frame_btn = PushButton(text="Next →")
        self.next_frame_btn.clicked.connect(self._next_frame)
        self.first_last_btn = PushButton(text="↕️ First/Last")
        self.first_last_btn.clicked.connect(self._jump_to_bounds)

        nav_layout.addWidget(self.prev_frame_btn.native)
        nav_layout.addWidget(self.next_frame_btn.native)
        nav_layout.addWidget(self.first_last_btn.native)
        layout.addLayout(nav_layout)

        # Separator
        layout.addWidget(Label(value="<hr>").native)

        # Tracking parameters section
        layout.addWidget(Label(value="<b>3. Tracking Parameters:</b>").native)

        # Shape type
        self.shape_type = ComboBox(
            label="Shape Type",
            choices=["rectangle", "disk"],
            value="rectangle",
        )
        layout.addWidget(self.shape_type.native)

        # Shape size
        self.shape_size = SpinBox(
            label="Shape Size", value=50, min=10, max=200, step=5
        )
        layout.addWidget(self.shape_size.native)

        # Grid size
        self.grid_size = SpinBox(
            label="Grid Size", value=5, min=3, max=15, step=1
        )
        layout.addWidget(self.grid_size.native)

        # Visualization options
        layout.addWidget(Label(value="<b>Visualization Options:</b>").native)

        vis_layout = QHBoxLayout()
        self.show_shapes = CheckBox(value=True, text="Show Shapes")
        self.show_centers = CheckBox(value=True, text="Show Centers")
        vis_layout.addWidget(self.show_shapes.native)
        vis_layout.addWidget(self.show_centers.native)
        layout.addLayout(vis_layout)

        # Separator
        layout.addWidget(Label(value="<hr>").native)

        # Tracking actions section
        layout.addWidget(Label(value="<b>4. Run Tracking:</b>").native)

        # Seed layers list
        self.seed_list = ComboBox(label="Active Seed Layers", choices=[])
        self.refresh_seeds_btn = PushButton(text="Refresh List")
        self.refresh_seeds_btn.clicked.connect(self._refresh_seed_list)

        seed_list_layout = QHBoxLayout()
        seed_list_layout.addWidget(self.seed_list.native)
        seed_list_layout.addWidget(self.refresh_seeds_btn.native)
        layout.addLayout(seed_list_layout)

        # Track single layer
        self.track_single_btn = PushButton(text="Track Selected Seed Layer")
        self.track_single_btn.clicked.connect(self._track_selected_layer)
        self.track_single_btn.enabled = False
        layout.addWidget(self.track_single_btn.native)

        # Track all layers
        self.track_all_btn = PushButton(text="Track All Seed Layers")
        self.track_all_btn.clicked.connect(self._track_all_layers)
        self.track_all_btn.enabled = False
        layout.addWidget(self.track_all_btn.native)

        # Status
        layout.addWidget(Label(value="<hr>").native)
        self.status_label = Label(
            value="<i>Ready. Add seed layers to begin.</i>"
        )
        layout.addWidget(self.status_label.native)

        # Info about saving
        layout.addWidget(
            Label(
                value="<i>Tip: Use File > Save Selected Layer(s) to export tracking results.</i>"
            ).native
        )

        # Connect viewer layer events
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)

        # Add stretch to push everything up
        layout.addStretch()

        # Auto-initialize if we have an image
        if self._get_image_layers():
            self.image_selector.value = self._get_image_layers()[0]

    def _pick_custom_color(self):
        """Open color picker dialog for custom color selection."""
        color = QColorDialog.getColor()
        if color.isValid():
            # Convert QColor to hex string
            hex_color = color.name()
            self.current_color = hex_color

            # Add to choices if not already there
            current_choices = list(self.color_selector.choices)
            if hex_color not in current_choices:
                current_choices.append(hex_color)
                self.color_selector.choices = current_choices

            # Set as current selection
            self.color_selector.value = hex_color
            show_info(f"Selected custom color: {hex_color}")

    def _get_next_color(self):
        """Get the next color from the default list."""
        color = self.default_colors[
            self.color_index % len(self.default_colors)
        ]
        self.color_index += 1
        return color

    def _get_image_layers(self):
        """Get list of image layer names."""
        import napari  # Import locally to avoid the NameError

        return [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Image)
        ]

    def _ensure_seed_manager_initialized(self):
        """Ensure seed manager is initialized."""
        if self.seed_manager is None:
            try:
                # Changed from utils.cot to .core
                self.seed_manager = SeedLayerManager(self.viewer)
                self.refresh_seeds_btn.enabled = True
                show_info("Seed manager initialized!")
                return True
            except (ImportError, AttributeError) as e:
                show_error(f"Error initializing seed manager: {str(e)}")
                return False
        return True

    def _ensure_image_loaded(self):
        """Ensure an image is loaded before tracking. Returns True if successful."""
        if self.images_rgb is not None:
            return True

        # Try to load the selected image
        if self.image_selector.value:
            show_info("Auto-loading selected image...")
            self._load_image()
            return self.images_rgb is not None
        else:
            show_error("No image layer selected!")
            return False

    def _connect_seed_layer_events(self, layer_name):
        """Connect to data change events for a seed layer."""
        import napari  # Import locally

        if layer_name in self.viewer.layers:
            layer = self.viewer.layers[layer_name]
            if isinstance(layer, napari.layers.Points):
                # Disconnect any existing connection
                if layer_name in self.layer_event_connections:
                    layer.events.data.disconnect(
                        self.layer_event_connections[layer_name]
                    )

                # Connect to data changes
                connection = layer.events.data.connect(
                    lambda e: self._check_enable_tracking()
                )
                self.layer_event_connections[layer_name] = connection

    def _disconnect_seed_layer_events(self, layer_name):
        """Disconnect data change events for a seed layer."""
        if layer_name in self.layer_event_connections:
            if layer_name in self.viewer.layers:
                layer = self.viewer.layers[layer_name]
                layer.events.data.disconnect(
                    self.layer_event_connections[layer_name]
                )
            del self.layer_event_connections[layer_name]

    def _on_layer_change(self, event):
        """Handle layer changes in viewer."""
        self.image_selector.choices = self._get_image_layers()
        if self.seed_manager:
            self._refresh_seed_list()
            # Check if tracking buttons should be enabled
            self._check_enable_tracking()

    def _check_enable_tracking(self):
        """Check if tracking buttons should be enabled."""
        if self.seed_manager is not None:
            # Check if we have seed layers with points
            seed_layers = self.seed_manager.get_all_seed_layers()
            has_points = False

            for layer_name in seed_layers:
                points = self.seed_manager.get_seed_points(layer_name)
                if points is not None and len(points) > 0:
                    has_points = True
                    # Also make sure we're connected to this layer's events
                    self._connect_seed_layer_events(layer_name)

            # Enable tracking buttons if we have points (image will be loaded automatically)
            if has_points and self._get_image_layers():
                self._enable_tracking_buttons()
                self.status_label.value = "<i>Ready to track! (Image will be loaded automatically)</i>"
            else:
                self.track_single_btn.enabled = False
                self.track_all_btn.enabled = False
                if not self._get_image_layers():
                    self.status_label.value = "<i>No image layers found.</i>"
                else:
                    self.status_label.value = (
                        "<i>Add points to seed layers to enable tracking.</i>"
                    )

    def _load_image(self):
        """Load selected image for tracking."""
        try:
            if not self.image_selector.value:
                show_error("Please select an image layer first!")
                return

            self.status_label.value = "<i>Loading image...</i>"
            images = get_image_from_layer(
                self.viewer, self.image_selector.value
            )
            self.images_rgb = prepare_images_for_tracking(images)

            # Update info
            shape_str = f"Shape: {self.images_rgb.shape} (T={self.images_rgb.shape[0]}, H={self.images_rgb.shape[1]}, W={self.images_rgb.shape[2]})"
            self.image_info.value = (
                f"<b>Loaded:</b> {self.image_selector.value} - {shape_str}"
            )
            self.status_label.value = "<i>Image loaded successfully!</i>"

            # Check if tracking should be enabled
            self._check_enable_tracking()

            show_info(
                f"Image '{self.image_selector.value}' loaded successfully!"
            )

        except (ValueError, KeyError, AttributeError) as e:
            show_error(f"Error loading image: {str(e)}")
            self.status_label.value = (
                f"<i style='color: red;'>Error: {str(e)}</i>"
            )

    def _add_seed_layer(self):
        """Add a new seed layer."""
        try:
            # Auto-initialize seed manager if needed
            if not self._ensure_seed_manager_initialized():
                return

            name = self.seed_name_input.value
            if not name:
                show_error("Please enter a seed layer name!")
                return

            # Get selected color
            color = self.color_selector.value

            # Use the modified add_seed_layer method that accepts color
            layer = self.seed_manager.add_seed_layer(name, color=color)

            # Connect to the new layer's events
            self._connect_seed_layer_events(layer.name)

            self._refresh_seed_list()

            # Update input for next layer
            seed_count = len(self.seed_manager.get_all_seed_layers())
            self.seed_name_input.value = f"SP{seed_count + 1}"

            # Update color selector to next default color
            next_color = self._get_next_color()
            self.color_selector.value = next_color

            show_info(f"Added seed layer '{layer.name}' with color {color}")

        except (ValueError, RuntimeError) as e:
            show_error(f"Error adding seed layer: {str(e)}")

    def _load_seed_from_csv(self):
        """Load seed points from CSV file."""
        try:
            # Auto-initialize seed manager if needed
            if not self._ensure_seed_manager_initialized():
                return

            # Open file dialog
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Load seed points from CSV",
                "",
                "CSV Files (*.csv);;All Files (*)",
            )

            if not filename:  # User cancelled
                return

            # Read the CSV file
            df = pd.read_csv(filename)

            # Check for required columns
            required_cols = ["axis-0", "axis-1", "axis-2"]  # napari format
            if not all(col in df.columns for col in required_cols):
                # Try alternate format
                if all(col in df.columns for col in ["t", "y", "x"]):
                    df = df.rename(
                        columns={"t": "axis-0", "y": "axis-1", "x": "axis-2"}
                    )
                else:
                    show_error(
                        "CSV must contain columns: axis-0, axis-1, axis-2 (or t, y, x)"
                    )
                    return

            # Extract points
            points = df[["axis-0", "axis-1", "axis-2"]].values

            # Get layer name from file or use default
            layer_name = os.path.splitext(os.path.basename(filename))[0]
            if not layer_name.startswith("seed_"):
                layer_name = f"seed_{layer_name}"

            # Check for color column or features
            color = self.color_selector.value  # default

            # Try to get color from features or color column
            if "features" in df.columns:
                try:
                    # napari sometimes stores color in features
                    import ast

                    features = ast.literal_eval(df["features"].iloc[0])
                    if isinstance(features, dict) and "face_color" in features:
                        color = features["face_color"]
                except (ValueError, SyntaxError):
                    pass
            elif "face_color" in df.columns:
                color = df["face_color"].iloc[0] if len(df) > 0 else color
            elif "color" in df.columns:
                color = df["color"].iloc[0] if len(df) > 0 else color

            # Add seed layer with points
            layer = self.seed_manager.add_seed_layer(
                layer_name.replace("seed_", ""), color=color
            )
            layer.data = points

            # Connect to events
            self._connect_seed_layer_events(layer.name)

            self._refresh_seed_list()

            # Update color selector for next layer
            next_color = self._get_next_color()
            self.color_selector.value = next_color

            show_info(f"Loaded {len(points)} points into layer '{layer.name}'")

        except (
            FileNotFoundError,
            pd.errors.ParserError,
            KeyError,
            ValueError,
        ) as e:
            show_error(f"Error loading seed points: {str(e)}")

    def _refresh_seed_list(self):
        """Refresh the seed layers list."""
        if self.seed_manager:
            self.seed_manager.refresh_layers()
            seed_layers = self.seed_manager.get_all_seed_layers()
            self.seed_list.choices = seed_layers

            if seed_layers:
                self.seed_list.value = seed_layers[0]
                # Connect to all seed layer events
                for layer_name in seed_layers:
                    self._connect_seed_layer_events(layer_name)

            # Always check if tracking should be enabled after refresh
            self._check_enable_tracking()

    def _enable_tracking_buttons(self):
        """Enable tracking buttons when ready."""
        self.track_single_btn.enabled = True
        self.track_all_btn.enabled = True

    # Time navigation methods
    def _get_current_points_layer(self):
        """Get the currently selected layer if it's a points layer."""
        if len(self.viewer.layers.selection) == 0:
            # Try to use the selected seed layer
            if (
                self.seed_list.value
                and self.seed_list.value in self.viewer.layers
            ):
                return self.viewer.layers[self.seed_list.value]
            show_error("No layer selected")
            return None

        # Get the first selected layer
        selected_layer = list(self.viewer.layers.selection)[0]

        if selected_layer._type_string != "points":
            show_error(
                f"Selected layer '{selected_layer.name}' is not a points layer"
            )
            return None

        if len(selected_layer.data) == 0:
            show_error(f"Points layer '{selected_layer.name}' has no points")
            return None

        return selected_layer

    def _previous_frame(self):
        """Jump to previous frame with points."""
        layer = self._get_current_points_layer()
        if layer is None:
            return

        current_t = self.viewer.dims.current_step[0]
        time_indices = np.unique(layer.data[:, 0].astype(int))
        prev_times = time_indices[time_indices < current_t]

        if len(prev_times) > 0:
            new_time = prev_times[-1]
            self.viewer.dims.current_step = (
                new_time,
                *self.viewer.dims.current_step[1:],
            )
            show_info(f"Jumped to frame {new_time}")
        else:
            show_info("No previous annotated frame")

    def _next_frame(self):
        """Jump to next frame with points."""
        layer = self._get_current_points_layer()
        if layer is None:
            return

        current_t = self.viewer.dims.current_step[0]
        time_indices = np.unique(layer.data[:, 0].astype(int))
        next_times = time_indices[time_indices > current_t]

        if len(next_times) > 0:
            new_time = next_times[0]
            self.viewer.dims.current_step = (
                new_time,
                *self.viewer.dims.current_step[1:],
            )
            show_info(f"Jumped to frame {new_time}")
        else:
            show_info("No next annotated frame")

    def _jump_to_bounds(self):
        """Toggle between first and last annotated frame."""
        layer = self._get_current_points_layer()
        if layer is None:
            return

        current_t = self.viewer.dims.current_step[0]
        time_indices = np.unique(layer.data[:, 0].astype(int))

        if current_t == time_indices[0]:
            # If at first, go to last
            new_time = time_indices[-1]
        else:
            # Otherwise, go to first
            new_time = time_indices[0]

        self.viewer.dims.current_step = (
            new_time,
            *self.viewer.dims.current_step[1:],
        )
        show_info(f"Jumped to frame {new_time}")

    def _track_selected_layer(self):
        """Track the selected seed layer."""
        try:
            if not self.seed_list.value:
                show_error("Please select a seed layer to track!")
                return

            # Auto-load image if needed
            if not self._ensure_image_loaded():
                return

            self.status_label.value = (
                f"<i>Tracking {self.seed_list.value}...</i>"
            )

            track_seed_layer(
                self.images_rgb,
                self.seed_manager,
                self.seed_list.value,
                shape_type=self.shape_type.value,
                shape_size=self.shape_size.value,
                grid_size=self.grid_size.value,
                show_shapes=self.show_shapes.value,
                show_centers=self.show_centers.value,
            )

            self.status_label.value = (
                f"<i>Tracking complete for {self.seed_list.value}!</i>"
            )
            show_info(f"Tracking complete for {self.seed_list.value}!")

        except (RuntimeError, ValueError, TypeError) as e:
            show_error(f"Error during tracking: {str(e)}")
            self.status_label.value = (
                f"<i style='color: red;'>Error: {str(e)}</i>"
            )

    def _track_all_layers(self):
        """Track all seed layers."""
        try:
            # Auto-load image if needed
            if not self._ensure_image_loaded():
                return

            seed_layers = self.seed_manager.get_all_seed_layers()
            if not seed_layers:
                show_error("No seed layers to track!")
                return

            self.status_label.value = (
                f"<i>Tracking {len(seed_layers)} seed layers...</i>"
            )

            track_all_seed_layers(
                self.images_rgb,
                self.seed_manager,
                shape_type=self.shape_type.value,
                shape_size=self.shape_size.value,
                grid_size=self.grid_size.value,
                show_shapes=self.show_shapes.value,
                show_centers=self.show_centers.value,
            )

            self.status_label.value = "<i>All tracking complete!</i>"
            show_info(
                f"Tracking complete for all {len(seed_layers)} seed layers!"
            )

        except (RuntimeError, ValueError, TypeError) as e:
            show_error(f"Error during tracking: {str(e)}")
            self.status_label.value = (
                f"<i style='color: red;'>Error: {str(e)}</i>"
            )


# Then create an alias with the exact name napari is looking for:
cotracker_widget = CoTrackerWidget
