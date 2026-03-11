"""
Ground-truth matcher for the Blender multi-object synthetic dataset.

Projects keypoints through each object's individual 3D transformation to
establish ground-truth correspondences, correctly handling scenes with
independently-moving objects.
"""

import logging

import numpy as np
import torch

from ...geometry.wrappers import Camera, Pose
from ...visualization.viz2d import (
    cm_RdGn,
    plot_images,
    plot_keypoints,
    plot_matches,
)
from ..base_model import BaseModel

logger = logging.getLogger(__name__)

# Compatibility shim for torch.amp.custom_fwd across PyTorch versions.
_AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)

# Colour palette for per-object visualisation.
DISTINCT_COLORS = [
    "cyan", "yellow", "lime", "magenta", "orange",
    "red", "blue", "green", "purple", "pink",
    "teal", "coral", "gold", "dodgerblue", "crimson",
]


class BlenderGTMatcher(BaseModel):
    """Object-aware ground-truth matcher.

    For every pair of keypoint sets it:

    1. Assigns each keypoint to an object via the segmentation mask.
    2. Projects keypoints from view 0 → view 1 (and vice-versa) using the
       per-object 3D transformations and depth maps.
    3. Finds mutual nearest-neighbour matches within a pixel threshold.
    """

    default_conf = {
        # Positive / negative match thresholds (pixels)
        "th_positive": 3.0,
        "th_negative": 5.0,
        "th_epi": None,
        "th_consistency": None,
        # Multi-object handling
        "match_same_object_only": True,
        "use_object_masks": True,
        "min_object_points": 5,
        # Bidirectional matching
        "use_bidirectional": True,
        "nn_threshold": 3.0,
        # Visualisation
        "visualize": False,
        "visualize_interval": 1,
    }

    required_data_keys = ["view0", "view1", "keypoints0", "keypoints1"]

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init(self, conf):
        self._batch_counter = 0
        if self.conf.use_object_masks:
            self.required_data_keys = list(self.required_data_keys) + [
                "view0.object_mask",
                "view1.object_mask",
            ]

    # ------------------------------------------------------------------
    # DictContainer unwrapping
    # ------------------------------------------------------------------

    @staticmethod
    def unwrap_dict_container(value, batch_idx=0):
        """Extract the underlying dict from a (possibly batched) DictContainer."""
        if isinstance(value, list):
            value = value[batch_idx]
        return value.data if hasattr(value, "data") else value

    # ------------------------------------------------------------------
    # Object-ID assignment
    # ------------------------------------------------------------------

    @staticmethod
    def get_object_ids_for_keypoints(keypoints, object_mask):
        """Assign each keypoint to an object ID via the segmentation mask.

        Args:
            keypoints: (N, 2) tensor of ``(x, y)`` pixel coordinates.
            object_mask: (H, W) tensor of integer object IDs.

        Returns:
            (N,) tensor of per-keypoint object IDs.
        """
        h, w = object_mask.shape
        uv = torch.round(keypoints).long()
        u = uv[:, 0].clamp(0, w - 1)
        v = uv[:, 1].clamp(0, h - 1)
        return object_mask[v, u]

    @staticmethod
    def get_object_ids_for_keypoints_batch(keypoints, object_masks):
        """Batched version of :meth:`get_object_ids_for_keypoints`.

        Args:
            keypoints: (B, N, 2) tensor.
            object_masks: (B, H, W) tensor.

        Returns:
            (B, N) tensor.
        """
        B, N, _ = keypoints.shape
        _, H, W = object_masks.shape
        uv = torch.round(keypoints).long()
        u = uv[..., 0].clamp(0, W - 1)
        v = uv[..., 1].clamp(0, H - 1)
        bi = torch.arange(B, device=keypoints.device).view(B, 1).expand(B, N)
        return object_masks[bi, v, u]

    # ------------------------------------------------------------------
    # Nearest-neighbour search
    # ------------------------------------------------------------------

    @staticmethod
    def find_nearest_neighbours(query, target, threshold=3.0):
        """Find the nearest target point for each query point.

        Args:
            query: (N, 2) projected locations.
            target: (M, 2) detected keypoints.
            threshold: maximum distance in pixels.

        Returns:
            indices: (N,) index into *target* (−1 if no neighbour within *threshold*).
            distances: (N,) distance to nearest target.
        """
        if len(query) == 0 or len(target) == 0:
            dev = query.device if len(query) else target.device
            return (
                torch.full((len(query),), -1, dtype=torch.long, device=dev),
                torch.full((len(query),), float("inf"), device=dev),
            )
        dists = torch.cdist(query.float(), target.float(), p=2)  # (N, M)
        min_d, min_idx = dists.min(dim=1)
        valid = min_d <= threshold
        indices = torch.where(
            valid, min_idx,
            torch.tensor(-1, dtype=torch.long, device=query.device),
        )
        return indices, min_d

    # ------------------------------------------------------------------
    # Depth reading
    # ------------------------------------------------------------------

    @staticmethod
    def read_depth_at(u, v, depth_map):
        """Read the depth value at pixel ``(u, v)`` using a 3×3 patch.

        Falls back to the minimum finite value in the patch to handle
        sub-pixel boundaries.

        Args:
            u, v: scalar tensors — pixel coordinates.
            depth_map: (H, W) tensor.

        Returns:
            float — depth value (``float('inf')`` if no valid value).
        """
        h, w = depth_map.shape
        ui, vi = int(round(u.item())), int(round(v.item()))
        x0, x1 = max(0, ui - 1), min(w - 1, ui + 1)
        y0, y1 = max(0, vi - 1), min(h - 1, vi + 1)
        patch = depth_map[y0 : y1 + 1, x0 : x1 + 1]
        finite = patch[torch.isfinite(patch)]
        if finite.numel() == 0:
            return float("inf")
        return finite.min().item()

    # ------------------------------------------------------------------
    # 3-D ↔ pixel conversions
    # ------------------------------------------------------------------

    @staticmethod
    def pixel_to_world(u, v, z_cam, K, R, t):
        """Back-project a pixel to a 3-D world point.

        Args:
            u, v: int — pixel coordinates.
            z_cam: scalar tensor — camera-space depth.
            K: (1, 3, 3) or (3, 3) intrinsic matrix.
            R: (1, 3, 3) or (3, 3) rotation (world→camera).
            t: (1, 3) or (3,) translation.

        Returns:
            (3,) world-space point tensor.
        """
        K_s = K.squeeze(0)
        R_s = R.squeeze(0)
        t_s = t.squeeze(0)
        uv1 = torch.tensor([u, v, 1.0], dtype=K.dtype, device=K.device)
        ray = torch.linalg.solve(K_s, uv1)
        X_cam = ray * z_cam
        return R_s.T @ (X_cam - t_s)

    @staticmethod
    def world_to_pixel(K, R, t, X_world):
        """Project a world point to pixel coordinates.

        Args:
            K: (1, 3, 3) or (3, 3) intrinsic matrix.
            R: (1, 3, 3) or (3, 3) rotation (world→camera).
            t: (1, 3) or (3,) translation.
            X_world: (3,) world point.

        Returns:
            uv: (2,) pixel coordinates.
            z_cam: scalar — depth in camera frame.
        """
        K_s = K.squeeze(0).float()
        R_s = R.squeeze(0).float()
        t_col = t.squeeze(0).reshape(3, 1).float()
        X_col = X_world.reshape(3, 1).float()
        X_cam = R_s @ X_col + t_col
        proj = K_s @ X_cam
        z_cam = proj[2, 0]
        u = proj[0, 0] / z_cam
        v = proj[1, 0] / z_cam
        return torch.stack([u, v]), z_cam

    # ------------------------------------------------------------------
    # Object-mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_same_object_mask(object_ids0, object_ids1):
        """Create a boolean mask indicating same-object keypoint pairs.

        Args:
            object_ids0: (B, N) tensor.
            object_ids1: (B, M) tensor.

        Returns:
            (B, N, M) boolean tensor (True where IDs match and are nonzero).
        """
        B, N = object_ids0.shape
        M = object_ids1.shape[1]
        ids0 = object_ids0.unsqueeze(2).expand(B, N, M)
        ids1 = object_ids1.unsqueeze(1).expand(B, N, M)
        return (ids0 == ids1) & (ids0 != 0)

    # ------------------------------------------------------------------
    # Per-object 4×4 matrix from Euler angles + translation + scale
    # ------------------------------------------------------------------

    @staticmethod
    def matrix_from_state(rot_euler, t, scale):
        """Build a 4×4 world matrix from Blender object state using pure PyTorch.

        Implements Translation @ Rotation_XYZ @ Scale, where Rotation_XYZ = Rz @ Ry @ Rx.

        Args:
            rot_euler: (3,) Euler angles in radians (XYZ order).
            t: (3,) world-space translation.
            scale: (3,) scale factors.

        Returns:
            (4, 4) float32 torch tensor.
        """
        # Ensure inputs are float32 tensors
        if not isinstance(rot_euler, torch.Tensor):
            rot_euler = torch.tensor(rot_euler, dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32)

        rot_euler = rot_euler.float()
        t = t.float()
        scale = scale.float()

        ax, ay, az = rot_euler.unbind(-1)
        cx, sx = torch.cos(ax), torch.sin(ax)
        cy, sy = torch.cos(ay), torch.sin(ay)
        cz, sz = torch.cos(az), torch.sin(az)

        sx_val, sy_val, sz_val = scale.unbind(-1)

        # R = Rz @ Ry @ Rx, combined with scale
        r00 = cy * cz * sx_val
        r10 = cy * sz * sx_val
        r20 = -sy * sx_val

        r01 = (cz * sy * sx - sz * cx) * sy_val
        r11 = (sz * sy * sx + cz * cx) * sy_val
        r21 = cy * sx * sy_val

        r02 = (cz * sy * cx + sz * sx) * sz_val
        r12 = (sz * sy * cx - cz * sx) * sz_val
        r22 = cy * cx * sz_val

        tx, ty, tz = t.unbind(-1)
        zeros = torch.zeros_like(tx)
        ones = torch.ones_like(tx)

        row0 = torch.stack([r00, r01, r02, tx], dim=-1)
        row1 = torch.stack([r10, r11, r12, ty], dim=-1)
        row2 = torch.stack([r20, r21, r22, tz], dim=-1)
        row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)

        return torch.stack([row0, row1, row2, row3], dim=-2)

    # ------------------------------------------------------------------
    # Occlusion check
    # ------------------------------------------------------------------

    def check_occlusion_batch(
        self, uv_batch, Z_batch, depth_map, tau_abs=1e-2, tau_rel=1e-2
    ):
        """Vectorised occlusion check using ``F.grid_sample``.

        A point is visible if the depth-buffer value at its projected pixel
        agrees with the projected depth within additive tolerances.

        Args:
            uv_batch: (K, 2) projected pixel coordinates.
            Z_batch: (K,) projected depths.
            depth_map: (H, W) depth buffer of the target view.

        Returns:
            (K,) boolean tensor — True if the point is *visible*.
        """
        H, W = depth_map.shape

        # Normalise coordinates to [-1, 1] for grid_sample
        grid = uv_batch.clone().float()
        grid[:, 0] = 2.0 * grid[:, 0] / (W - 1) - 1.0
        grid[:, 1] = 2.0 * grid[:, 1] / (H - 1) - 1.0
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, K, 2)

        depth_4d = depth_map.unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        sampled = torch.nn.functional.grid_sample(
            depth_4d, grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )
        z_buf = sampled.squeeze()  # (K,)

        valid_depth = torch.isfinite(z_buf) & (z_buf > 0)
        tau = tau_abs + tau_rel * z_buf
        not_occluded = torch.abs(Z_batch - z_buf) <= tau

        return valid_depth & not_occluded

    # ------------------------------------------------------------------
    # Vectorised per-object projection (view_src → view_dst)
    # ------------------------------------------------------------------

    def map_all_points_vectorized(
        self,
        keypoints,
        object_ids,
        view_src,
        view_dst,
        batch_idx,
        check_occlusion=False,
    ):
        """Project keypoints from *view_src* to *view_dst* using per-object
        3-D transformations.  Fully vectorised — all keypoints are processed
        in a single batched pass per object.

        Args:
            keypoints: (N, 2) source keypoint coordinates.
            object_ids: (N,) per-keypoint object IDs.
            view_src / view_dst: dicts with ``camera``, ``depth``,
                ``objects_stats`` keys.
            batch_idx: batch element index (for unwrapping DictContainers).
            check_occlusion: if True, mark occluded projections as invalid.

        Returns:
            projected: (N, 2) projected pixel coordinates in *view_dst*.
            valid: (N,) boolean mask of valid projections.
        """
        dev = keypoints.device
        N = len(keypoints)

        cam_src = self.unwrap_dict_container(view_src["camera"], batch_idx)
        cam_dst = self.unwrap_dict_container(view_dst["camera"], batch_idx)
        stats_src = self.unwrap_dict_container(view_src["objects_stats"], batch_idx)
        stats_dst = self.unwrap_dict_container(view_dst["objects_stats"], batch_idx)

        depth_src = view_src["depth"]
        if depth_src.dim() == 3:
            depth_src = depth_src[batch_idx]
        depth_dst = view_dst["depth"]
        if depth_dst.dim() == 3:
            depth_dst = depth_dst[batch_idx]

        H, W = depth_src.shape

        unique_objs = object_ids.unique()
        unique_objs = unique_objs[unique_objs != 0]

        if len(unique_objs) == 0:
            return (
                torch.full((N, 2), float("nan"), device=dev),
                torch.zeros(N, dtype=torch.bool, device=dev),
            )

        # Pre-compute per-object transform matrices T = M_dst @ inv(M_src)
        T_matrices = {}
        for obj_id_tensor in unique_objs:
            obj_id = str(obj_id_tensor.item())
            if obj_id not in stats_src or obj_id not in stats_dst:
                continue
            obj0 = stats_src[obj_id]
            obj1 = stats_dst[obj_id]
            M0 = self.matrix_from_state(obj0["rot_euler"], obj0["t"], obj0["scale"]).to(dev)
            M1 = self.matrix_from_state(obj1["rot_euler"], obj1["t"], obj1["scale"]).to(dev)
            T_matrices[obj_id_tensor.item()] = (M1 @ torch.linalg.inv(M0)).float()

        # Camera matrices
        K0 = torch.tensor(cam_src["K"], dtype=torch.float32, device=dev).squeeze()
        R0 = torch.tensor(cam_src["R"], dtype=torch.float32, device=dev).squeeze()
        t0 = torch.tensor(cam_src["t"], dtype=torch.float32, device=dev).squeeze()
        K1 = torch.tensor(cam_dst["K"], dtype=torch.float32, device=dev).squeeze()
        R1 = torch.tensor(cam_dst["R"], dtype=torch.float32, device=dev).squeeze()
        t1 = torch.tensor(cam_dst["t"], dtype=torch.float32, device=dev).squeeze().reshape(3, 1)

        # Batch pixel -> world for ALL keypoints at once
        uv_rounded = torch.round(keypoints).long()
        u_all = uv_rounded[:, 0].clamp(0, W - 1)
        v_all = uv_rounded[:, 1].clamp(0, H - 1)
        Z0_all = depth_src[v_all, u_all]  # (N,)

        ones = torch.ones(N, 1, device=dev, dtype=torch.float32)
        uv_hom = torch.cat([keypoints.float(), ones], dim=1)  # (N, 3)
        x_c_dir = torch.linalg.solve(K0, uv_hom.T).T  # (N, 3)
        X_c = x_c_dir * Z0_all.unsqueeze(1)  # (N, 3)
        X_w = (X_c - t0) @ R0  # (N, 3) — world coords in src frame

        # Apply per-object transforms
        X_w_hom = torch.cat([X_w, ones], dim=1)  # (N, 4)
        X_w1 = torch.full_like(X_w, float("nan"))
        for obj_id_tensor in unique_objs:
            obj_id_int = obj_id_tensor.item()
            if obj_id_int not in T_matrices:
                continue
            obj_mask = (object_ids == obj_id_tensor) & torch.isfinite(Z0_all) & (Z0_all > 0)
            T = T_matrices[obj_id_int]
            X_w1[obj_mask] = (T @ X_w_hom[obj_mask].T).T[:, :3]

        # Batch world -> pixel in dst
        X_c1 = (R1 @ X_w1.T + t1).T  # (N, 3)
        x_proj = (K1 @ X_c1.T).T      # (N, 3)
        Z1 = x_proj[:, 2]
        uv1 = x_proj[:, :2] / Z1.unsqueeze(1)  # (N, 2)

        # Validity
        valid_transform = ~torch.isnan(X_w1[:, 0])
        valid_bounds = (
            (uv1[:, 0] >= 0) & (uv1[:, 0] < W) &
            (uv1[:, 1] >= 0) & (uv1[:, 1] < H) &
            (Z1 > 0)
        )
        valid = valid_transform & valid_bounds

        if check_occlusion:
            valid_occ = self.check_occlusion_batch(
                uv1, Z1, depth_dst
            )
            valid = valid & valid_occ

        uv_proj = torch.full((N, 2), float("nan"), device=dev)
        uv_proj[valid] = uv1[valid]

        return uv_proj, valid

    # ------------------------------------------------------------------
    # Match assignment helpers
    # ------------------------------------------------------------------

    def matches_to_assignment(self, matches0, matches1):
        """Convert batched match vectors to a batched assignment matrix.

        Args:
            matches0: (B, N0) indices into view 1 (−1 for unmatched).
            matches1: (B, N1) indices into view 0 (−1 for unmatched).

        Returns:
            (B, N0, N1) boolean assignment tensor.
        """
        B, N0 = matches0.shape
        N1 = matches1.shape[1]
        dev = matches0.device

        assignment = torch.zeros(B, N0, N1, dtype=torch.bool, device=dev)
        valid_forward = matches0 >= 0

        batch_idx = torch.arange(B, device=dev).view(B, 1).expand(B, N0)
        kp0_idx = torch.arange(N0, device=dev).view(1, N0).expand(B, N0)

        matches0_clamped = torch.clamp(matches0, min=0)
        backward_matches = matches1[batch_idx, matches0_clamped]  # (B, N0)
        bidirectional = (backward_matches == kp0_idx) & valid_forward

        b_indices = batch_idx[bidirectional]
        i_indices = kp0_idx[bidirectional]
        j_indices = matches0[bidirectional]
        assignment[b_indices, i_indices, j_indices] = True

        return assignment

    # ------------------------------------------------------------------
    # Bidirectional matching
    # ------------------------------------------------------------------

    def match_bidirectional_batch(
        self, kp0_batch, kp1_batch, obj_ids0_batch, obj_ids1_batch,
        view0, view1,
    ):
        """Run bidirectional matching for each element in a batch.

        Returns:
            matches0: (B, N) tensor.
            matches1: (B, M) tensor.
            scores0: (B, N) tensor of reprojection distances (inf if unmatched).
        """
        B = kp0_batch.shape[0]
        N = kp0_batch.shape[1]
        M = kp1_batch.shape[1]
        dev = kp0_batch.device

        all_m0 = torch.full((B, N), -1, dtype=torch.long, device=dev)
        all_m1 = torch.full((B, M), -1, dtype=torch.long, device=dev)
        all_s0 = torch.full((B, N), float("inf"), device=dev)

        for b in range(B):
            m0, m1, s0 = self.match_bidirectional(
                kp0_batch[b], kp1_batch[b],
                obj_ids0_batch[b], obj_ids1_batch[b],
                view0, view1, batch_idx=b,
            )
            all_m0[b] = m0
            all_m1[b] = m1
            all_s0[b] = s0

        return all_m0, all_m1, all_s0

    def match_bidirectional(
        self, kp0, kp1, obj_ids0, obj_ids1,
        view0, view1, batch_idx,
    ):
        """Compute mutual-nearest-neighbour GT matches between two views.

        Projects kp0 → view1 (forward) and kp1 → view0 (backward), finds
        nearest neighbours in both directions, and retains only mutual
        matches.

        Returns:
            matches0: (N,) indices into kp1 (−1 if unmatched).
            matches1: (M,) indices into kp0 (−1 if unmatched).
        """
        N, M = kp0.shape[0], kp1.shape[0]
        dev = kp0.device
        th = self.conf.th_positive

        map_v0 = {
            "camera": view0["camera"],
            "depth": view0["depth"],
            "objects_stats": view0["objects_stats"],
        }
        map_v1 = {
            "camera": view1["camera"],
            "depth": view1["depth"],
            "objects_stats": view1["objects_stats"],
        }

        proj_01, valid_01 = self.map_all_points_vectorized(
            kp0, obj_ids0, map_v0, map_v1,
            batch_idx=batch_idx, check_occlusion=True,
        )
        proj_10, valid_10 = self.map_all_points_vectorized(
            kp1, obj_ids1, map_v1, map_v0,
            batch_idx=batch_idx, check_occlusion=True,
        )

        matches0 = torch.full((N,), -1, dtype=torch.long, device=dev)
        matches1 = torch.full((M,), -1, dtype=torch.long, device=dev)
        scores0 = torch.full((N,), float("inf"), device=dev)

        # Per-object matching: only match keypoints belonging to the same
        # object, then map local indices back to global positions.
        unique_objs = obj_ids0.unique()
        unique_objs = unique_objs[unique_objs != 0]  # skip background

        for obj_id in unique_objs:
            # Forward: kp0 on this object -> nearest kp1 on same object
            mask0_fwd = (obj_ids0 == obj_id) & valid_01
            mask1_fwd = (obj_ids1 == obj_id)
            if mask0_fwd.sum() > 0 and mask1_fwd.sum() > 0:
                kp0_idx = torch.where(mask0_fwd)[0]
                kp1_idx = torch.where(mask1_fwd)[0]
                nn_idx, nn_dist = self.find_nearest_neighbours(
                    proj_01[mask0_fwd], kp1[mask1_fwd], threshold=th
                )
                for ii, (ni, nd) in enumerate(zip(nn_idx, nn_dist)):
                    if ni >= 0:
                        matches0[kp0_idx[ii]] = kp1_idx[ni]
                        scores0[kp0_idx[ii]] = nd

            # Backward: kp1 on this object -> nearest kp0 on same object
            mask1_bwd = (obj_ids1 == obj_id) & valid_10
            mask0_bwd = (obj_ids0 == obj_id)
            if mask1_bwd.sum() > 0 and mask0_bwd.sum() > 0:
                kp1_idx = torch.where(mask1_bwd)[0]
                kp0_idx = torch.where(mask0_bwd)[0]
                nn_idx, nn_dist = self.find_nearest_neighbours(
                    proj_10[mask1_bwd], kp0[mask0_bwd], threshold=th
                )
                for ii, (ni, nd) in enumerate(zip(nn_idx, nn_dist)):
                    if ni >= 0:
                        matches1[kp1_idx[ii]] = kp0_idx[ni]

        # Enforce mutual consistency
        if self.conf.use_bidirectional:
            for i in range(N):
                j = matches0[i].item()
                if j >= 0 and matches1[j].item() != i:
                    matches0[i] = -1
                    scores0[i] = float("inf")
            for j in range(M):
                i = matches1[j].item()
                if i >= 0 and matches0[i].item() != j:
                    matches1[j] = -1

        return matches0, matches1, scores0

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @_AMP_CUSTOM_FWD_F32
    def _forward(self, data):
        B = data["keypoints0"].shape[0]
        N = data["keypoints0"].shape[1]
        M = data["keypoints1"].shape[1]
        dev = data["keypoints0"].device

        obj_mask0 = data["view0"]["object_mask"]
        obj_mask1 = data["view1"]["object_mask"]

        if obj_mask0.dim() == 3 and obj_mask0.shape[0] == B:
            pass  # already (B, H, W)
        elif obj_mask0.dim() == 3:
            obj_mask0 = obj_mask0.squeeze(0) if B == 1 else obj_mask0
            obj_mask1 = obj_mask1.squeeze(0) if B == 1 else obj_mask1

        obj_ids0 = self.get_object_ids_for_keypoints_batch(
            data["keypoints0"],
            obj_mask0.unsqueeze(0) if obj_mask0.dim() == 2 else obj_mask0,
        )
        obj_ids1 = self.get_object_ids_for_keypoints_batch(
            data["keypoints1"],
            obj_mask1.unsqueeze(0) if obj_mask1.dim() == 2 else obj_mask1,
        )

        matches0, matches1, scores0 = self.match_bidirectional_batch(
            data["keypoints0"], data["keypoints1"],
            obj_ids0, obj_ids1,
            data["view0"], data["view1"],
        )

        if self.conf.visualize:
            self._batch_counter += 1
            if self._batch_counter % self.conf.visualize_interval == 0:
                self._visualize_batch_sample(data, matches0, obj_ids0)

        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": scores0,
            "matching_scores1": (matches1 > -1).float(),
            "assignment": self.matches_to_assignment(matches0, matches1),
        }

    # ------------------------------------------------------------------
    # Visualisation helpers
    # ------------------------------------------------------------------

    def _visualize_batch_sample(self, data, matches0_batch, obj_ids0_batch, batch_idx=0):
        kp0 = data["keypoints0"][batch_idx].cpu()
        kp1 = data["keypoints1"][batch_idx].cpu()
        m0 = matches0_batch[batch_idx].cpu()
        oids = obj_ids0_batch[batch_idx].cpu()

        img0 = data["view0"]["image"][batch_idx].permute(1, 2, 0).cpu().numpy()
        img1 = data["view1"]["image"][batch_idx].permute(1, 2, 0).cpu().numpy()

        valid = m0 > -1
        mkp0 = kp0[valid]
        mkp1 = kp1[m0[valid]]
        n_matches = int(valid.sum())

        logger.info("Visualising %d GT matches", n_matches)
        plot_images([img0, img1], titles=["View 0", "View 1"])
        plot_keypoints([kp0, kp1], ps=3)
        plot_matches(mkp0, mkp1, color="lime", lw=0.5)

    def _visualize_gt_matches(self, data, gt_01, object_ids0, b_idx):
        img0 = data["view0"]["image"][b_idx].permute(1, 2, 0).cpu().numpy()
        img1 = data["view1"]["image"][b_idx].permute(1, 2, 0).cpu().numpy()
        kp0 = data["keypoints0"][b_idx].cpu()
        kp1 = data["keypoints1"][b_idx].cpu()

        plot_images([img0, img1], titles=["View 0 – GT projected", "View 1"])
        plot_keypoints([kp0, kp1], ps=3)

        valid = gt_01[:, 0] >= 0
        if valid.any():
            src = kp0[gt_01[valid, 0].long()]
            dst = kp1[gt_01[valid, 1].long()]
            plot_matches(src, dst, color="cyan", lw=0.5)

    # ------------------------------------------------------------------
    # Loss (not used during evaluation)
    # ------------------------------------------------------------------

    def loss(self, pred, data):
        raise NotImplementedError("BlenderGTMatcher is evaluation-only.")