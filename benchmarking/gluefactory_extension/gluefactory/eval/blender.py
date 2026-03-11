"""
Evaluation pipeline for the Blender multi-object synthetic dataset.

Metrics
-------
- Number of matches and keypoints
- Reprojection-based match precision (1 px, 3 px, 5 px) using ground-truth depth
- Ground-truth match recall and precision (per-object-aware via BlenderGTMatcher)
- Relative-pose estimation AUC (5°, 10°, 20°) via RANSAC
"""

import h5py
import logging
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..geometry.wrappers import Camera, Pose
from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..visualization.viz2d import plot_cumulative
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_poses, eval_relative_pose_robust

logger = logging.getLogger(__name__)

# Optional per-object ground-truth matcher.
try:
    from ..models.matchers.blender_dataset_matcher import BlenderGTMatcher

    _HAS_GT_MATCHER = True
except ImportError:
    _HAS_GT_MATCHER = False
    logger.warning(
        "BlenderGTMatcher not available (missing mathutils?). "
        "GT match precision / recall metrics will be skipped."
    )


# ---------------------------------------------------------------------------
# Camera / Pose helpers
# ---------------------------------------------------------------------------


def _camera_from_dict(cam_data, image_w, image_h):
    """Build a :class:`Camera` from raw Blender camera NPZ data.

    Supports two storage conventions:
    - ``K`` key: a 3x3 intrinsic matrix.
    - ``f_x``, ``f_y``, ``c_x``, ``c_y`` keys: individual components.
    """
    if "K" in cam_data:
        K = np.asarray(cam_data["K"], dtype=np.float64)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    else:
        fx = float(cam_data["f_x"])
        fy = float(cam_data["f_y"])
        cx = float(cam_data["c_x"])
        cy = float(cam_data["c_y"])

    tensor = torch.tensor(
        [image_w, image_h, fx, fy, cx, cy], dtype=torch.float32
    )
    return Camera(tensor)


def _pose_from_dict(cam_data):
    """Build a world-to-camera :class:`Pose` from raw NPZ data."""
    R = torch.tensor(cam_data["R"], dtype=torch.float32)
    t = torch.tensor(cam_data["t"], dtype=torch.float32).flatten()
    return Pose.from_Rt(R, t)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def prepare_blender_data(data):
    """Convert raw Blender camera / depth data to the format expected by the
    standard evaluation utilities (:class:`Camera`, :class:`Pose`, resized
    depth).  Modifies *data* in-place and returns it.
    """
    poses = {}
    for view_key in ("view0", "view1"):
        view = data[view_key]

        # Unwrap DictContainer (may be list after collation).
        cam_container = view["camera"]
        if isinstance(cam_container, list):
            cam_data = cam_container[0].data
        else:
            cam_data = cam_container.data

        # Original image size — stored by ImagePreprocessor as [w, h].
        orig_size = view["original_image_size"]
        if isinstance(orig_size, torch.Tensor):
            if orig_size.dim() > 1:
                orig_w, orig_h = orig_size[0].float().tolist()
            else:
                orig_w, orig_h = orig_size.float().tolist()
        else:
            orig_w, orig_h = float(orig_size[0]), float(orig_size[1])

        camera = _camera_from_dict(cam_data, orig_w, orig_h)

        # Scale camera to the preprocessed (resized) resolution.
        scales = view["scales"]
        if isinstance(scales, torch.Tensor) and scales.dim() > 1:
            scales = scales[0]
        camera = camera.scale(scales)

        # Add batch dimension and replace camera in-place.
        view["camera"] = Camera(camera._data.unsqueeze(0))

        # Extrinsics (world-to-camera).
        poses[view_key] = _pose_from_dict(cam_data)

        # Depth: convert to tensor and resize to match image if needed.
        depth = view.get("depth")
        if depth is not None:
            if not isinstance(depth, torch.Tensor):
                depth = torch.tensor(depth, dtype=torch.float32)
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)

            img_tensor = view.get("image")
            if img_tensor is not None:
                target_h, target_w = img_tensor.shape[-2:]
                depth_h, depth_w = depth.shape[-2:]
                if depth_h != target_h or depth_w != target_w:
                    depth = F.interpolate(
                        depth.unsqueeze(0).float(),
                        size=(target_h, target_w),
                        mode="nearest",
                    ).squeeze(0)
            view["depth"] = depth

        # Object mask: convert to tensor and resize to match image.
        obj_mask = view.get("object_mask")
        if obj_mask is not None:
            if not isinstance(obj_mask, torch.Tensor):
                obj_mask = torch.tensor(obj_mask, dtype=torch.long)
            if obj_mask.dim() == 2:
                obj_mask = obj_mask.unsqueeze(0)

            img_tensor = view.get("image")
            if img_tensor is not None:
                target_h, target_w = img_tensor.shape[-2:]
                mask_h, mask_w = obj_mask.shape[-2:]
                if mask_h != target_h or mask_w != target_w:
                    obj_mask = F.interpolate(
                        obj_mask.unsqueeze(0).float(),
                        size=(target_h, target_w),
                        mode="nearest",
                    ).squeeze(0).long()
            view["object_mask"] = obj_mask

    # Relative pose with batch dimension.
    T_0to1 = poses["view1"] @ poses["view0"].inv()
    data["T_0to1"] = Pose(T_0to1._data.unsqueeze(0))

    return data


# ---------------------------------------------------------------------------
# Per-object GT match evaluation
# ---------------------------------------------------------------------------


def _eval_gt_matches(data, pred, raw_cameras, gt_matcher):
    """Compute GT match recall and precision using the per-object-aware
    :class:`BlenderGTMatcher`.

    Returns a dict with ``gt_match_recall@3px`` and
    ``gt_match_precision@3px``.
    """
    matcher_input = {
        "keypoints0": pred["keypoints0"].unsqueeze(0),
        "keypoints1": pred["keypoints1"].unsqueeze(0),
        "view0": {
            "camera": raw_cameras["view0"],
            "depth": data["view0"]["depth"],
            "object_mask": data["view0"]["object_mask"],
            "objects_stats": data["view0"]["objects_stats"],
        },
        "view1": {
            "camera": raw_cameras["view1"],
            "depth": data["view1"]["depth"],
            "object_mask": data["view1"]["object_mask"],
            "objects_stats": data["view1"]["objects_stats"],
        },
    }

    with torch.no_grad():
        gt_pred = gt_matcher(matcher_input)

    gt_m0 = gt_pred["matches0"]              # (1, N0)
    pred_m0 = pred["matches0"].unsqueeze(0)   # (1, N0)

    # Recall: among GT-positive pairs, how many did the model find?
    gt_pos = (gt_m0 > -1).float()
    recall = ((pred_m0 == gt_m0) * gt_pos).sum(1) / (1e-8 + gt_pos.sum(1))

    # Precision: among predicted matches, how many agree with GT?
    pred_pos = ((pred_m0 > -1) & (gt_m0 >= -1)).float()
    precision = ((pred_m0 == gt_m0) * pred_pos).sum(1) / (
        1e-8 + pred_pos.sum(1)
    )

    return {
        "gt_match_recall@3px": recall[0].item(),
        "gt_match_precision@3px": precision[0].item(),
    }


# ---------------------------------------------------------------------------
# Per-keypoint object-category helpers
# ---------------------------------------------------------------------------


def _keypoint_object_ids(kpts, obj_mask):
    """Map keypoints to object IDs via the per-pixel object mask.

    Args:
        kpts: (N, 2) tensor of (x, y) pixel coordinates.
        obj_mask: (1, H, W) or (H, W) long tensor of object IDs.

    Returns:
        (N,) long tensor of object IDs.
    """
    mask = obj_mask[0] if obj_mask.dim() == 3 else obj_mask
    h, w = mask.shape
    x = kpts[:, 0].long().clamp(0, w - 1)
    y = kpts[:, 1].long().clamp(0, h - 1)
    return mask[y, x]


def _compute_per_keypoint_data(data, pred, raw_cameras, gt_matcher, eval_conf):
    """Compute per-keypoint GT matches, object IDs, and reprojection errors.

    Uses :class:`BlenderGTMatcher` for object-aware GT matching and
    per-object-transform reprojection — which is correct even for dynamic
    (moved) objects.

    Returns:
        dict of numpy arrays with keypoints, predicted / GT matches,
        object IDs, and reprojection errors per view.
    """
    kp0 = pred["keypoints0"]  # (N0, 2)
    kp1 = pred["keypoints1"]  # (N1, 2)
    N0, N1 = len(kp0), len(kp1)

    obj_mask0 = data["view0"]["object_mask"]
    obj_mask1 = data["view1"]["object_mask"]

    obj_ids0 = _keypoint_object_ids(kp0, obj_mask0)
    obj_ids1 = _keypoint_object_ids(kp1, obj_mask1)

    # GT matches via BlenderGTMatcher (per-object-aware)
    matcher_data = {
        "keypoints0": kp0.unsqueeze(0),
        "keypoints1": kp1.unsqueeze(0),
        "view0": {
            "camera": raw_cameras["view0"],
            "depth": data["view0"]["depth"],
            "object_mask": obj_mask0,
            "objects_stats": data["view0"]["objects_stats"],
        },
        "view1": {
            "camera": raw_cameras["view1"],
            "depth": data["view1"]["depth"],
            "object_mask": obj_mask1,
            "objects_stats": data["view1"]["objects_stats"],
        },
    }

    with torch.no_grad():
        gt_pred = gt_matcher(matcher_data)

    gt_m0 = gt_pred["matches0"][0]  # (N0,)
    gt_m1 = gt_pred["matches1"][0]  # (N1,)

    # Per-object-aware reprojection error for PREDICTED matches
    map_view0 = {
        "camera": raw_cameras["view0"],
        "depth": data["view0"]["depth"],
        "objects_stats": data["view0"]["objects_stats"],
    }
    map_view1 = {
        "camera": raw_cameras["view1"],
        "depth": data["view1"]["depth"],
        "objects_stats": data["view1"]["objects_stats"],
    }

    # Forward: project all kp0 -> view1
    proj_01, valid_01 = gt_matcher.map_all_points_vectorized(
        kp0, obj_ids0, map_view0, map_view1,
        batch_idx=0, check_occlusion=False,
    )
    # Backward: project all kp1 -> view0
    proj_10, valid_10 = gt_matcher.map_all_points_vectorized(
        kp1, obj_ids1, map_view1, map_view0,
        batch_idx=0, check_occlusion=False,
    )

    pred_m0 = pred["matches0"]
    reproj_error0 = np.full(N0, np.nan, dtype=np.float32)
    match_mask0 = pred_m0 > -1
    if match_mask0.any():
        j_idx = pred_m0[match_mask0].long()
        fwd = (proj_01[match_mask0] - kp1[j_idx]).norm(dim=-1)
        bwd = (proj_10[j_idx] - kp0[match_mask0]).norm(dim=-1)
        both_valid = valid_01[match_mask0] & valid_10[j_idx]
        sym_err = 0.5 * (fwd + bwd)
        sym_err[~both_valid] = float("nan")
        reproj_error0[match_mask0.cpu().numpy()] = (
            sym_err.cpu().detach().numpy()
        )

    pred_m1 = pred["matches1"]
    reproj_error1 = np.full(N1, np.nan, dtype=np.float32)
    match_mask1 = pred_m1 > -1
    if match_mask1.any():
        i_idx = pred_m1[match_mask1].long()
        fwd = (proj_10[match_mask1] - kp0[i_idx]).norm(dim=-1)
        bwd = (proj_01[i_idx] - kp1[match_mask1]).norm(dim=-1)
        both_valid = valid_10[match_mask1] & valid_01[i_idx]
        sym_err = 0.5 * (fwd + bwd)
        sym_err[~both_valid] = float("nan")
        reproj_error1[match_mask1.cpu().numpy()] = (
            sym_err.cpu().detach().numpy()
        )

    return {
        "keypoints0": kp0.cpu().numpy().astype(np.float32),
        "keypoints1": kp1.cpu().numpy().astype(np.float32),
        "pred_matches0": pred_m0.cpu().numpy().astype(np.int32),
        "pred_matches1": pred_m1.cpu().numpy().astype(np.int32),
        "gt_matches0": gt_m0.cpu().numpy().astype(np.int32),
        "gt_matches1": gt_m1.cpu().numpy().astype(np.int32),
        "object_ids0": obj_ids0.cpu().numpy().astype(np.int32),
        "object_ids1": obj_ids1.cpu().numpy().astype(np.int32),
        "reproj_error0": reproj_error0,
        "reproj_error1": reproj_error1,
    }


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------


class BlenderPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "blender",
            "batch_size": 1,
            "num_workers": 8,
            "preprocessing": {
                "resize": 1024,
                "side": "long",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,
            },
        },
        "eval": {
            "estimator": "opencv",
            "ransac_th": 1.0,
            "room_object_ids": [1, 2, 3, 4, 5, 6],
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
        "matches0",
        "matches1",
        "matching_scores0",
        "matching_scores1",
    ]
    optional_export_keys = []

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the full evaluation loop and return aggregated metrics."""
        assert pred_file.exists()
        conf = self.conf.eval
        results = defaultdict(list)

        test_thresholds = (
            (
                [conf.ransac_th]
                if conf.ransac_th > 0
                else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            )
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )

        pose_results = defaultdict(lambda: defaultdict(list))
        cache_loader = CacheLoader(
            {"path": str(pred_file), "collate": None}
        ).eval()

        # Per-object GT matcher for match precision / recall.
        gt_matcher = None
        if _HAS_GT_MATCHER:
            try:
                gt_matcher = BlenderGTMatcher(
                    {"visualize": False, "nn_threshold": 3.0}
                ).eval()
            except Exception as e:
                logger.warning(
                    "Could not instantiate BlenderGTMatcher: %s", e
                )

        # Per-keypoint output file.
        per_kp_path = pred_file.parent / "per_keypoint.h5"
        h5_per_kp = h5py.File(str(per_kp_path), "w")

        for i, data in enumerate(tqdm(loader)):
            pred = cache_loader(data)

            # Save raw camera DictContainers before prepare_blender_data
            # replaces them with Camera wrapper objects.
            raw_cameras = {}
            for vk in ("view0", "view1"):
                raw_cameras[vk] = data[vk]["camera"]

            try:
                data = prepare_blender_data(data)
            except Exception as e:
                logger.warning(
                    "Skipping sample %d: could not prepare camera data (%s)",
                    i, e,
                )
                continue

            results_i = {}

            # --- GT match precision / recall (per-object aware) --------------
            has_depth = (
                "depth" in data["view0"]
                and data["view0"]["depth"] is not None
                and "depth" in data["view1"]
                and data["view1"]["depth"] is not None
            )
            has_masks = (
                "object_mask" in data["view0"]
                and data["view0"]["object_mask"] is not None
                and "object_mask" in data["view1"]
                and data["view1"]["object_mask"] is not None
                and "objects_stats" in data["view0"]
                and "objects_stats" in data["view1"]
            )
            if gt_matcher is not None and has_masks and has_depth:
                try:
                    gt_match_results = _eval_gt_matches(
                        data, pred, raw_cameras, gt_matcher
                    )
                    results_i.update(gt_match_results)
                except Exception as e:
                    logger.debug(
                        "GT match eval failed for sample %d: %s", i, e
                    )

            # --- Relative pose estimation via RANSAC -------------------------
            for th in test_thresholds:
                try:
                    pose_results_i = eval_relative_pose_robust(
                        data,
                        pred,
                        {"estimator": conf.estimator, "ransac_th": th},
                    )
                    for k, v in pose_results_i.items():
                        pose_results[th][k].append(v)
                except Exception as e:
                    logger.debug(
                        "Pose eval failed for sample %d, th=%s: %s", i, th, e
                    )
                    pose_results[th]["rel_pose_error"].append(float("inf"))
                    pose_results[th]["ransac_inl"].append(0)
                    pose_results[th]["ransac_inl%"].append(0)

            # --- Per-keypoint data (per-object-aware) ------------------------
            if gt_matcher is not None and has_masks and has_depth:
                try:
                    per_kp = _compute_per_keypoint_data(
                        data, pred, raw_cameras, gt_matcher, conf
                    )
                    pair_name = data["name"][0]
                    grp = h5_per_kp.create_group(pair_name)
                    for k, v in per_kp.items():
                        grp.create_dataset(k, data=v)
                except Exception as e:
                    logger.debug(
                        "Per-keypoint data failed for sample %d: %s", i, e
                    )

            # --- Book-keeping ------------------------------------------------
            results_i["names"] = data["name"][0]
            if "scene" in data:
                results_i["scenes"] = data["scene"][0]

            for k, v in results_i.items():
                results[k].append(v)

        # Close per-keypoint file.
        h5_per_kp.close()
        logger.info("Per-keypoint data written to %s", per_kp_path)

        # --- Aggregate -------------------------------------------------------
        summaries = {}
        for k, v in results.items():
            arr = np.array(v)
            if not np.issubdtype(arr.dtype, np.number):
                continue
            summaries[f"m{k}"] = round(np.mean(arr), 3)

        # Pose AUC.
        auc_ths = [5, 10, 20]
        best_pose_results, best_th = eval_poses(
            pose_results, auc_ths=auc_ths, key="rel_pose_error"
        )
        results = {**results, **pose_results[best_th]}
        summaries = {**summaries, **best_pose_results}

        figures = {
            "pose_recall": plot_cumulative(
                {self.conf.eval.estimator: results["rel_pose_error"]},
                [0, 30],
                unit="°",
                title="Pose ",
            ),
        }

        return summaries, figures, results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .. import logger  # noqa: F811

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(BlenderPipeline.default_conf)

    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name, args, "configs/", default_conf
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = BlenderPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()