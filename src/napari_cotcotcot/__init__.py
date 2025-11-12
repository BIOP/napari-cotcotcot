"""napari-cotcotcot: A napari plugin for CoTracker-based tracking."""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._widget import cotracker_widget

__all__ = ("cotracker_widget",)

def main():
    import napari
    
    # Create the napari viewer
    viewer = napari.Viewer()
    
    # Import and open your widget
    # from napari_cotcotcot import CoTrackerWidget  # Replace with actual widget
    from ._widget import CoTrackerWidget

    # Add your widget to the viewer
    # Method 1: If your widget is a dock widget
    viewer.window.add_dock_widget(CoTrackerWidget(viewer), name='CotCotCot')

    # Method 2: If you're using the napari plugin system
    # viewer.window._activate_plugin_dock_widget('napari-cotcotcot', 'Your Widget Name')
    
    # Start the napari event loop
    napari.run()

if __name__ == "__main__":
    main()
