
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from skimage.io import imread
from napari.types import LayerData

def _load_seed_from_csv_standalone(csv_path: str) -> np.ndarray:
    """
    Load seed points from a CSV file.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing seed point coordinates.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) containing seed point coordinates.

    Raises
    ------
    ValueError
        If the CSV format is invalid (missing required columns).
    """
    df = pd.read_csv(csv_path)

    required_cols = ["axis-0", "axis-1", "axis-2"]
    if not all(col in df.columns for col in required_cols):
        if all(col in df.columns for col in ["t", "y", "x"]):
            df = df.rename(columns={"t": "axis-0", "y": "axis-1", "x": "axis-2"})
        else:
            raise ValueError("CSV must contain columns: axis-0, axis-1, axis-2 (or t, y, x)")

    return df[["axis-0", "axis-1", "axis-2"]].values

def load_sample_data() -> List[LayerData]:
    """
    Loads sample data for demonstration in napari.

    This function loads:
        - A GIF image ("chicken-run.gif") from a remote URL:
          https://raw.githubusercontent.com/BIOP/napari-cotcotcot/refs/heads/main/src/napari_cotcotcot/data/Gallus_gallus_domesticus/chicken-run.gif
        - Seed points from a local CSV file ("seed_Chicken1.csv") if available at:
          <package_dir>/data/Gallus_gallus_domesticus/seed_Chicken1.csv

    Returns
    -------
    List[LayerData]
        A list of tuples (data, metadata, layer_type) suitable for napari, where:
            - The first tuple contains the image data, metadata with name "cotcotcot", and layer_type "image".
            - The second tuple (if CSV is found and loaded) contains the seed points, metadata with name "seed_Chicken1", and layer_type "points".

    Data Sources
    -----------
    - GIF image: downloaded from the above URL.
    - Seed points: loaded from a local CSV file.

    Exceptions
    ----------
    - If the CSV file is missing, empty, or malformed, an error is printed and only the image is returned.
    - If the GIF image cannot be downloaded or read, an exception may be raised by skimage.io.imread.
    """
    URL = "https://raw.githubusercontent.com/BIOP/napari-cotcotcot/refs/heads/main/src/napari_cotcotcot/data/Gallus_gallus_domesticus/chicken-run.gif"
    print(f"Downloading sample data from {URL}...")
    gif_data = imread(URL)
    print("Downloading completed!")

    sample_datasets = [(gif_data, {"name": "cotcotcot"}, "image")]

    # Load CSV
    csv_data_path = Path(__file__).parent / "data" / "Gallus_gallus_domesticus" / "seed_Chicken1.csv"
    if csv_data_path.exists():
        try:
            csv_data = _load_seed_from_csv_standalone(str(csv_data_path))
            sample_datasets.append((csv_data, {"name": "seed_Chicken1"}, "points"))
        except (FileNotFoundError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"Error loading CSV data: {e}")

    return sample_datasets
