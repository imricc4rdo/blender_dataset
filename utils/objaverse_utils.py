from pathlib import Path
from typing import List, Optional
import objaverse.xl as oxl
import pandas as pd


def get_annotations(annotation_folder: Path) -> pd.DataFrame:
    """
    Retrieves and filters Objaverse annotations for .glb files not sourced from GitHub.
    
    :param annotation_folder: Path to the folder where annotations are stored.
    :type annotation_folder: Path
    
    :return annotations: Filtered annotations DataFrame.
    :rtype: pandas.DataFrame
    """
    # Get annotations
    annotations = oxl.get_annotations(download_dir = annotation_folder)
    # Filter for .glb files not from GitHub
    annotations = annotations.loc[
        (annotations['fileType'] == 'glb') &
        (annotations['source'] != 'github')
    ].reset_index(drop = True)
    return annotations


def pick_objects(n_add: int, annotations: pd.DataFrame, placed_objs: List[str], download_dir: Path, 
                 processes: Optional[int], size_mb_threshold: float) -> List[Path]:
    """
    Picks and downloads a specified number of unique 3D objects from Objaverse annotations.
    
    :param n_add: Number of objects to add.
    :type n_add: int
    :param annotations: Objaverse annotations DataFrame.
    :type annotations: pd.DataFrame
    :param placed_objs: List of already placed object names.
    :type placed_objs: List[str]
    :param download_dir: Directory to download objects to.
    :type download_dir: Path
    :param processes: Number of processes to use for downloading.
    :type processes: Optional[int]
    :param size_mb_threshold: Maximum allowed size of downloaded objects in megabytes.
    :type size_mb_threshold: float
    
    :return found_objects: List of paths to the downloaded objects.
    :rtype: List[Path]
    """
    # Ensure download directory exists
    download_dir.mkdir(parents = True, exist_ok = True)

    # Validate annotations
    if len(annotations) == 0:
        raise RuntimeError('[WARN] No objects found with the given filters.')

    # Pick and download objects
    tries, max_tries = 0, n_add * 10
    found_objects = []
    while len(found_objects) < n_add and tries < max_tries:
        tries += 1
        n_left = n_add - len(found_objects)
        
        # Choose random object(s)
        picked = annotations.sample(n = n_left)

        # Download object(s)
        try:
            downloaded = oxl.download_objects(
                picked,
                download_dir = download_dir,
                processes = processes,
            )
            
            # Check if objects were already selected
            for identifier, obj_path in downloaded.items():
                p = Path(obj_path)
                
                # Check file size
                size_mb = p.stat().st_size / (1024 * 1024)
                if size_mb > size_mb_threshold:
                    # Ignore objects exceeding size limit
                    print(f'[SIZE CHECK] Skipping {p.name}: {size_mb:.1f}MB exceeds {size_mb_threshold}MB limit')
                    continue
                
                # Add unique objects only
                if p not in found_objects and p.stem not in placed_objs:
                    found_objects.append(p)
        
        # Handle exceptions
        except Exception as e:
            print(f'[DOWNLOAD] {type(e).__name__}: {e}')
            print('[INFO] Skipping this batch and retrying...')
    
    # Check if enough objects were found
    if len(found_objects) < n_add:
        raise RuntimeError(f'[WARN] Only found {len(found_objects)} objects after {tries} attempts.')

    return found_objects