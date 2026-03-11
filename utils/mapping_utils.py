import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
from pathlib import Path
from mathutils import Matrix, Euler, Vector


# --- 2D_0 → 3D_0 ---

def load_exr(path: str | Path, dtype = np.float32) -> np.ndarray:
    """
    Load a single-channel EXR file exported from Blender.
    
    :param path: Path to the EXR file.
    :type path: str or Path
    :param dtype: Desired data type of the output array (default: np.float32).
    :type dtype: data-type, optional
    
    :return: 2D array containing the EXR data.
    :rtype: np.ndarray
    """
    EXR = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    if EXR is None:
        raise FileNotFoundError(f'[WARN] cv2.imread failed (file missing or unreadable): {path}')

    # Convert to float32 array
    EXR = np.asarray(EXR, dtype = dtype)

    # Handle EXR with channels (H, W, 1): take the first channel
    if EXR.ndim == 3 and EXR.shape[-1] >= 1:
        EXR = EXR[..., 0]

    return np.ascontiguousarray(EXR)


def depth_min3x3(D: np.ndarray) -> np.ndarray:
    """
    Compute the minimum depth value in a 3x3 neighborhood around each pixel.
    
    :param D: Input depth map
    :type D: np.ndarray
    
    :return: Depth map where each pixel contains the minimum depth in its 3x3 neighborhood
    :rtype: np.ndarray
    """
    D = D.astype(np.float32, copy = False)
    big = np.float32(1e10)

    # Replace non-finite values (NaN, inf) with a large sentinel value
    D_safe = np.where(np.isfinite(D), D, big)

    # Apply morphological erosion to get minimum depth in 3x3 window
    kernel = np.ones((3, 3), np.uint8)
    D_min = cv2.erode(D_safe, kernel, iterations = 1)

    # Restore NaN for pixels that had sentinel values
    D_min[D_min >= big * 0.5] = np.nan
    return D_min


def pixel_to_world(u: float, v: float, Zc: float, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Convert a pixel (u, v) with depth Zc into a 3D point in world coordinates.
    
    :param u: Horizontal pixel coordinate (column index).
    :type u: float
    :param v: Vertical pixel coordinate (row index).
    :type v: float
    :param Zc: Depth value at the pixel (distance from the camera in meters).
    :type Zc: float
    :param K: Intrinsic matrix of the camera (3x3).
    :type K: np.ndarray
    :param R: Rotation matrix of the camera (3x3, OpenCV convention).
    :type R: np.ndarray
    :param t: Translation vector of the camera (3,).
    :type t: np.ndarray
        
    :return: 3D point in world coordinates corresponding to the input pixel and depth.
    :rtype: np.ndarray
    """
    # Pixel in homogeneous form
    p = np.array([u, v, 1.0], dtype = np.float64)

    # Convert pixel → normalized camera ray (z = 1)
    x_c_dir = np.linalg.inv(K) @ p

    # Scale by actual depth to get camera-space coordinates
    X_c = x_c_dir * Zc

    # Convert camera → world 
    X_w = R.T @ (X_c - t)

    return X_w


# --- 3D_0 → 3D_1 ---

def matrix_from_state(rot_euler: np.ndarray, t: np.ndarray, scale: np.ndarray) -> Matrix:
    """
    Construct a 4x4 world transformation matrix from Euler rotation,
    translation, and scale components -> M_world = T * R * S.
    
    :param rot_euler: Rotation angles (in radians) for X, Y, Z, in Blender’s Euler order.
    :type rot_euler: array-like of length 3
    :param t: Translation vector (x, y, z) in world units.
    :type t: array-like of length 3
    :param scale: Non-uniform object scale along x, y, z.
    :type scale: array-like of length 3

    :return: 4x4 transformation matrix combining translation, rotation, and scale.
    :rtype: mathutils.Matrix
    """
    # Rotation matrix (convert Euler angles → 3×3 → 4×4)
    R = Euler(rot_euler).to_matrix().to_4x4()

    # Scale matrix: diagonal matrix with scale factors
    S = Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))

    # Translation matrix: moves the object in world space
    T = Matrix.Translation(Vector(t))

    # Combine in the correct order: T * R * S
    return T @ R @ S


def compute_transform_0_to_1(data0: dict, data1: dict) -> Matrix:
    """
    Compute the 4x4 transformation that maps points from
    configuration 0 to configuration 1 of the same object.
    If:
        M0 maps object-local → world coordinates in view 0,
        M1 maps object-local → world coordinates in view 1,
    Then the mapping world0 → world1 is:
        T01 = M1 * inverse(M0)
        
    :param data0: Loaded .npz data containing the transformation components for view 0.
    :type data0: dict-like
    :param data1: Loaded .npz data containing the transformation components for view 1.
    :type data1: dict-like
    
    :return: Transformation matrix that converts a point from world0 to world1.
    :rtype: mathutils.Matrix
    """
    # Construct world matrices for view 0 and view 1
    M0 = matrix_from_state(data0['rot_euler'], data0['t'], data0['scale'])
    M1 = matrix_from_state(data1['rot_euler'], data1['t'], data1['scale'])
    
    # Compute world_0 → world_1 transformation
    T01 = M1 @ M0.inverted()
    return T01


def apply_transform(Xw0: np.ndarray, T01: Matrix) -> Vector:
    """
    Apply the 4x4 transformation T01 to a 3D point expressed in the
    world coordinates of view 0, producing the corresponding point
    in world coordinates of view 1.
    
    :param Xw0: 3D point in world coordinates of view 0.
    :type Xw0: array-like of length 3
    :param T01: Transformation matrix mapping world0 → world1.
    :type T01: mathutils.Matrix
    
    :return: Transformed 3D point in world coordinates of view 1.
    :rtype: mathutils.Vector
    """
    # Convert the 3D point (Xw), given as a NumPy array, into a Blender Vector
    p0 = Vector(Xw0)

    # Convert point into homogeneous coordinates (x, y, z, 1)
    p4 = Vector((*p0, 1.0))

    # Multiply by the affine transformation and return xyz part
    return list((T01 @ p4).xyz)


# --- 3D_1 → 2D_1 ---

def world_to_pixel(K: np.ndarray, R: np.ndarray, t: np.ndarray, Xw: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Project a 3D world-space point into 2D pixel coordinates.
    
    :param K: Intrinsic matrix of the camera (3x3).
    :type K: np.ndarray
    :param R: Rotation matrix of the camera (3x3, OpenCV convention).
    :type R: np.ndarray
    :param t: Translation vector of the camera (3,).
    :type t: np.ndarray
    :param Xw: 3D point in world coordinates.
    :type Xw: array-like of length 3
    
    :return: A tuple containing: pixel coordinates corresponding to the 3D point and the depth value in camera coordinates.
    :rtype: tuple[np.ndarray, float]
    """
    # Ensure proper shapes: Xw and t become column vectors (3x1)
    Xw = np.asarray(Xw).reshape(3, 1)
    t = np.asarray(t).reshape(3, 1)

    # Convert world → camera
    Xc = R @ Xw + t

    # Convert camera → pixel
    x = K @ Xc          
    u = x[0, 0] / x[2, 0]
    v = x[1, 0] / x[2, 0]
    
    return np.array([u, v]), x[2, 0]


# --- Mapping ---

def map_points(obj_pass_idx_str: str, coords: np.ndarray, depth0: np.ndarray, depth1_occ: np.ndarray, poses0: dict, poses1: dict, 
               camera0: dict, camera1: dict, check_occlusion: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Map multiple 2D pixel coordinates from view0 to view1.
    
    :param obj_pass_idx_str: Pass index as string (object ID).
    :type obj_pass_idx_str: str
    :param coords: Array of (v, u) pixel coordinates to map.
    :type coords: ndarray of shape (N, 2)
    :param depth0: Depth map of view0.
    :type depth0: ndarray of shape (H, W)
    :param depth1_occ: Depth map of view1 with minimum depth in a 3x3 neighborhood around each pixel.
    :type depth1_occ: ndarray of shape (H, W)
    :param poses0: Poses dictionary for view0.
    :type poses0: dict
    :param poses1: Poses dictionary for view1.
    :type poses1: dict
    :param camera0: Camera parameters for view0.
    :type camera0: dict
    :param camera1: Camera parameters for view1.
    :type camera1: dict
    :param check_occlusion: Whether to check occlusion in target view (default: True).
    :type check_occlusion: bool, optional
    
    :return: A tuple containing: boolean array indicating which input coordinates were successfully mapped and 
    array of (u, v) pixel coordinates in view1 corresponding to the input coordinates (undefined for invalid entries).
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    N = len(coords)
    if N == 0:
        return np.array([], dtype = bool), np.zeros((0, 2))
    
    H, W = depth0.shape
    
    # Extract u, v coordinates
    v_coords = coords[:, 0]  # Row indices
    u_coords = coords[:, 1]  # Column indices
    
    # --- 2D_0 -> 3D_0 ---
    # Unpack camera parameters for view0
    K0, R0, t0 = camera0['K'], camera0['R'], camera0['t']
    
    # Get depth values for all pixels
    Z = depth0[v_coords, u_coords]  # (N,)
    
    # Convert to homogeneous pixel coordinates
    pixels = np.stack([u_coords, v_coords, np.ones(N)], axis = 1)  # (N, 3)
    
    # Unproject to camera space
    K0_inv = np.linalg.inv(K0)
    X_c = (K0_inv @ pixels.T).T * Z[:, np.newaxis]  # (N, 3)
    
    # Transform to world space
    X_w0 = (R0.T @ (X_c - t0).T).T  # (N, 3)
    
    
    # --- 3D_0 -> 3D_1 ---
    # Load object transformations
    obj_data0 = poses0[obj_pass_idx_str]
    
    if obj_pass_idx_str not in poses1:
        return np.zeros(N, dtype = bool), np.zeros((N, 2))
    
    obj_data1 = poses1[obj_pass_idx_str]
    
    # Compute transformation matrix (same for all points)
    T01 = compute_transform_0_to_1(obj_data0, obj_data1)
    T01_np = np.array(T01)  # Convert to numpy 4x4
    
    # Transform all points at once
    X_w0_h = np.concatenate([X_w0, np.ones((N, 1))], axis = 1)  # (N, 4) homogeneous
    X_w1_h = (T01_np @ X_w0_h.T).T  # (N, 4)
    X_w1 = X_w1_h[:, :3]  # (N, 3)
    
    
    # --- 3D_1 -> 2D_1 ---
    # Unpack camera parameters for view1
    K1, R1, t1 = camera1['K'], camera1['R'], camera1['t']
    
    # Transform to camera space
    X_c1 = (R1 @ X_w1.T).T + t1  # (N, 3)
    
    # Project to image plane
    X_proj = (K1 @ X_c1.T).T  # (N, 3)
    u_mapped = X_proj[:, 0] / X_proj[:, 2]
    v_mapped = X_proj[:, 1] / X_proj[:, 2]
    z_c = X_proj[:, 2]
    
    # Check bounds using integer coordinates
    u_int = np.rint(u_mapped).astype(np.int32)
    v_int = np.rint(v_mapped).astype(np.int32)
    valid_mask = (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    
    
    # --- Occlusion check ---
    if check_occlusion:
        # Get valid points
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Get depth buffer values for all valid points at once
            u_valid = u_int[valid_indices]
            v_valid = v_int[valid_indices]
            z_buf = depth1_occ[v_valid, u_valid]
            z_c_valid = z_c[valid_indices]
            
            # Check occlusion for all points
            tau_abs = 1e-2
            tau_rel = 1e-2
            tau = tau_abs + tau_rel * z_buf
            
            # Mark as invalid if depth check fails or z_buf is nan
            depth_check_failed = np.isnan(z_buf) | (np.abs(z_c_valid - z_buf) > tau)
            
            # Update valid_mask
            invalid_global_indices = valid_indices[depth_check_failed]
            valid_mask[invalid_global_indices] = False
    
    mapped_coords = np.stack([u_int, v_int], axis = 1)  # (N, 2)
    
    return valid_mask, mapped_coords