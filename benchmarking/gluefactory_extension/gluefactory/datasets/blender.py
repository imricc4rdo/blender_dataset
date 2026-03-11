"""
Dataset loader for the Blender multi-object synthetic dataset.

Loads rendered images, depth maps, object masks, and camera / object
transformation data for pairs or triplets of views.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility wrapper
# ---------------------------------------------------------------------------

class DictContainer:
    """Thin wrapper that prevents ``collate_fn`` from merging dictionaries.

    Camera parameters and per-object statistics may have different keys
    across samples (e.g. different visible object IDs).  Wrapping them in
    a ``DictContainer`` causes the default collate to collect them into a
    plain Python list instead of attempting a recursive merge.
    """

    def __init__(self, data: dict):
        self.data = data

    # No-op so that ``batch_to_device`` does not fail — the contained
    # numpy arrays are not tensors and do not need to move to GPU.
    def to(self, *args, **kwargs):
        return self

    def __repr__(self):
        return f"DictContainer({self.data})"


# ---------------------------------------------------------------------------
# Top-level dataset (dispatches to Pair / Triplet variant)
# ---------------------------------------------------------------------------

class BlenderDataset(BaseDataset):
    """Blender multi-object synthetic dataset.

    Depending on ``conf.views`` (2 or 3), the underlying
    :class:`PairMultiObject` or :class:`TripletMultiObject` is used.
    """

    default_conf = {
        "data_dir": "blender_dataset",
        "train_split": "train_scenes.txt",
        "val_split": "val_scenes.txt",
        "val_pairs": None,
        "test_split": "test_scenes.txt",
        "views": 2,
        "train_num_per_scene": None,
        "val_num_per_scene": None,
        "test_num_per_scene": None,
        "grayscale": False,
        "read_depth": True,
        "read_image": True,
        "read_object_masks": True,
        "batch_size": 1,
        "num_workers": 1,
        "prefetch_factor": 1,
        "seed": 0,
        "reseed": False,
        "preprocessing": {
            "resize": 1024,
            "side": "long",
            "square_pad": False,
        },
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        assert self.conf.views in [1, 2, 3]
        if self.conf.views == 3:
            return TripletMultiObject(self.conf, split)
        return PairMultiObject(self.conf, split)


# ---------------------------------------------------------------------------
# Pair dataset
# ---------------------------------------------------------------------------

class PairMultiObject(torch.utils.data.Dataset):
    """Yields pairs of views from Blender multi-object scenes."""

    def __init__(self, conf, split):
        self.conf = conf
        self.root = DATA_PATH / conf.data_dir
        self.split = split
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        logger.info("Loading %s dataset from %s", split, self.root)

        self.scenes = self._load_scene_list(split)
        self.items = []
        self.sample_new_items(conf.seed)

        logger.info(
            "Loaded %d scenes with %d pairs for %s",
            len(self.scenes), len(self.items), split,
        )

    # ---- scene discovery --------------------------------------------------

    def _load_scene_list(self, split):
        """Load scene names from the split file.

        Supports both a flat layout (scenes directly under ``data_dir``)
        and a nested layout with grouping folders (e.g. ``3/scene_0001``).
        Falls back to auto-discovery if the split file does not exist.
        """
        split_file = self.root / self.conf[f"{split}_split"]

        if not split_file.exists():
            logger.warning(
                "Split file %s not found — auto-discovering scenes", split_file
            )
            scenes = []
            for entry in sorted(self.root.iterdir()):
                if entry.is_dir() and any(entry.glob("render*.png")):
                    scenes.append(entry.name)
                elif entry.is_dir():
                    # Nested layout: look one level deeper
                    for sub in sorted(entry.iterdir()):
                        if sub.is_dir() and any(sub.glob("render*.png")):
                            scenes.append(f"{entry.name}/{sub.name}")
            return scenes

        with open(split_file, "r") as fh:
            return [line.strip() for line in fh if line.strip()]

    # ---- item sampling ----------------------------------------------------

    def sample_new_items(self, seed):
        """Generate all view pairs for every scene, then shuffle."""
        logger.info("Sampling new %s pairs with seed %d", self.split, seed)
        self.items = []

        num_per_scene = self.conf[f"{self.split}_num_per_scene"]

        for scene_name in self.scenes:
            scene_dir = self.root / scene_name
            if not scene_dir.exists():
                logger.warning("Scene directory not found: %s", scene_dir)
                continue

            render_files = sorted(scene_dir.glob("render*.png"))
            n_views = len(render_files)
            if n_views == 0:
                logger.warning("No render files in %s", scene_dir)
                continue

            # All ordered pairs (i, j) with i < j
            pairs = [
                (scene_name, i, j)
                for i in range(n_views)
                for j in range(i + 1, n_views)
            ]

            if num_per_scene is not None and len(pairs) > num_per_scene:
                rng = np.random.RandomState(seed)
                idx = rng.choice(len(pairs), num_per_scene, replace=False)
                pairs = [pairs[k] for k in idx]

            self.items.extend(pairs)

        np.random.RandomState(seed).shuffle(self.items)

    # ---- I/O helpers ------------------------------------------------------

    @staticmethod
    def _load_image(scene_dir, view_idx, grayscale=False):
        path = scene_dir / f"render{view_idx}.png"
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return load_image(path, grayscale=grayscale)

    @staticmethod
    def _load_depth(scene_dir, view_idx):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        path = scene_dir / f"depth-{view_idx:04d}.exr"
        if not path.exists():
            return None
        depth = cv2.imread(
            str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
        ).astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth

    @staticmethod
    def _load_object_mask(scene_dir, view_idx):
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        path = scene_dir / f"obj_mask_for_view-{view_idx:04d}.exr"
        if not path.exists():
            return None
        mask = cv2.imread(
            str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR
        ).astype(np.int32)
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    # ---- single-view reader -----------------------------------------------

    def _read_view(self, scene_dir, view_idx):
        """Read and preprocess all data for one view."""
        # Image
        if self.conf.read_image:
            img = self._load_image(scene_dir, view_idx, self.conf.grayscale)
        else:
            ch = 1 if self.conf.grayscale else 3
            img = torch.zeros([ch, 512, 512], dtype=torch.float32)

        data = self.preprocessor(img)

        # Object mask
        if self.conf.read_object_masks:
            raw_mask = self._load_object_mask(scene_dir, view_idx)
            if raw_mask is not None:
                scales = data["scales"]
                new_h = int(raw_mask.shape[0] * scales[1])
                new_w = int(raw_mask.shape[1] * scales[0])
                resized = cv2.resize(
                    raw_mask.astype(np.float32),
                    (new_w, new_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
                data["object_mask"] = torch.from_numpy(resized).unsqueeze(0)

        # Depth
        if self.conf.read_depth:
            raw_depth = self._load_depth(scene_dir, view_idx)
            if raw_depth is not None:
                scales = data["scales"]
                new_h = int(raw_depth.shape[0] * scales[1])
                new_w = int(raw_depth.shape[1] * scales[0])
                resized = cv2.resize(
                    raw_depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST
                )
                data["depth"] = torch.from_numpy(resized).unsqueeze(0)

        # Camera intrinsics / extrinsics (wrapped to avoid collate merging)
        cam_path = scene_dir / f"camera{view_idx}.npz"
        cam_data = dict(np.load(cam_path))
        data["camera"] = DictContainer(cam_data)

        # Per-object poses
        poses_path = scene_dir / f"objs_per_view_{view_idx}.npz"
        raw_poses = np.load(poses_path, allow_pickle=True)["poses"].item()

        poses_dict = {}
        for obj_id, obj_data in raw_poses.items():
            poses_dict[obj_id] = {
                "rot_euler": torch.tensor(
                    np.array(obj_data["rot_euler"]), dtype=torch.float32
                ),
                "t": torch.tensor(
                    np.array(obj_data["t"]), dtype=torch.float32
                ),
                "scale": torch.tensor(
                    np.array(obj_data["scale"]), dtype=torch.float32
                ),
            }
        data["objects_stats"] = DictContainer(poses_dict)

        return data

    # ---- __getitem__ ------------------------------------------------------

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        return self.getitem(idx)

    def getitem(self, idx):
        scene_name, idx0, idx1 = self.items[idx]
        scene_dir = self.root / scene_name

        data0 = self._read_view(scene_dir, idx0)
        data1 = self._read_view(scene_dir, idx1)

        data = {
            "view0": data0,
            "view1": data1,
            "name": f"{scene_name}_{idx0}_{idx1}",
            "scene": scene_name,
            "idx": idx,
        }

        # Common objects visible in both views
        objs0 = set(data0["objects_stats"].data.keys())
        objs1 = set(data1["objects_stats"].data.keys())
        data["common_objects"] = list(objs0 & objs1)

        return data

    def __len__(self):
        return len(self.items)


# ---------------------------------------------------------------------------
# Triplet dataset
# ---------------------------------------------------------------------------

class TripletMultiObject(PairMultiObject):
    """Yields triplets of views from Blender multi-object scenes."""

    def sample_new_items(self, seed):
        logger.info("Sampling new %s triplets with seed %d", self.split, seed)
        self.items = []

        num_per_scene = self.conf[f"{self.split}_num_per_scene"]

        for scene_name in self.scenes:
            scene_dir = self.root / scene_name
            if not scene_dir.exists():
                logger.warning("Scene directory not found: %s", scene_dir)
                continue

            render_files = sorted(scene_dir.glob("render*.png"))
            n_views = len(render_files)
            if n_views < 3:
                logger.warning("Fewer than 3 views in %s", scene_dir)
                continue

            triplets = [
                (scene_name, i, j, k)
                for i in range(n_views)
                for j in range(i + 1, n_views)
                for k in range(j + 1, n_views)
            ]

            if num_per_scene is not None and len(triplets) > num_per_scene:
                rng = np.random.RandomState(seed)
                sel = rng.choice(len(triplets), num_per_scene, replace=False)
                triplets = [triplets[s] for s in sel]

            self.items.extend(triplets)

        np.random.RandomState(seed).shuffle(self.items)

    def getitem(self, idx):
        scene_name, idx0, idx1, idx2 = self.items[idx]
        scene_dir = self.root / scene_name

        data0 = self._read_view(scene_dir, idx0)
        data1 = self._read_view(scene_dir, idx1)
        data2 = self._read_view(scene_dir, idx2)

        data = {
            "view0": data0,
            "view1": data1,
            "view2": data2,
            "name": f"{scene_name}_{idx0}_{idx1}_{idx2}",
            "scene": scene_name,
            "idx": idx,
        }

        objs0 = set(data0["objects_stats"].data.keys())
        objs1 = set(data1["objects_stats"].data.keys())
        objs2 = set(data2["objects_stats"].data.keys())
        data["common_objects_01"] = list(objs0 & objs1)
        data["common_objects_02"] = list(objs0 & objs2)
        data["common_objects_12"] = list(objs1 & objs2)

        return data