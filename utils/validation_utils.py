import bpy
import numpy as np
from PIL import Image
from pathlib import Path
from pygltflib import GLTF2
from typing import List, Tuple
from utils.mapping_utils import load_exr, depth_min3x3, map_points
from scipy.stats import entropy as calculate_entropy


# --- INTEGRITY VALIDATION ---

class IntegrityValidationError(Exception):
    """Raised when an imported object fails integrity checks."""
    pass

def validate_object(obj_path: Path) -> None:
    """
    Validates the integrity of the object at obj_path.
    
    :param obj_path: Path to the object file.
    :type obj_path: Path
    """
    
    # Check if file exists
    if not obj_path.exists():
        raise IntegrityValidationError(f'[INTEGRITY CHECK] "{obj_path.stem}": File does not exist')
    
    # Check if file is a valid glTF
    try:
        gltf = GLTF2().load(obj_path)
    except Exception as e:
        raise IntegrityValidationError(f'[INTEGRITY CHECK] "{obj_path.stem}": Failed to load glTF - {e}')
    
    # Check if meshes are present
    if not gltf.meshes:
        raise IntegrityValidationError(f'[INTEGRITY CHECK] "{obj_path.stem}": No meshes found in glTF')
    
    # Check for NaN/Inf in transform values
    if gltf.nodes:
        for node in gltf.nodes:
            # Check translation
            if node.translation:
                if not all(np.isfinite(v) for v in node.translation):
                    raise IntegrityValidationError(f'[INTEGRITY CHECK] "{obj_path.stem}": Non-finite translation values')
            # Check rotation (quaternion)
            if node.rotation:
                if not all(np.isfinite(v) for v in node.rotation):
                    raise IntegrityValidationError(f'[INTEGRITY CHECK] "{obj_path.stem}": Non-finite rotation values')
            # Check scale
            if node.scale:
                if not all(np.isfinite(v) for v in node.scale):
                    raise IntegrityValidationError(f'[INTEGRITY CHECK] "{obj_path.stem}": Non-finite scale values')
                # Scale values - check for near-zero or negative
                if any(abs(v) < 1e-6 for v in node.scale):
                    raise IntegrityValidationError(f'[INTEGRITY CHECK] "{obj_path.stem}": Near-zero scale values')
    
    # Check for NaN/Inf in vertex coordinates
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            if primitive.attributes.POSITION is not None:
                accessor = gltf.accessors[primitive.attributes.POSITION]
                # Check min/max bounds if available (optional in glTF spec)
                if accessor.min and accessor.max:
                    if not all(np.isfinite(v) for v in accessor.min + accessor.max):
                        raise IntegrityValidationError(f'[INTEGRITY CHECK] "{obj_path.stem}": Non-finite vertex coordinates in bounds')


# --- COMPLEXITY VALIDATION ---

class ComplexityValidationError(Exception):
    """Raised when an imported object fails complexity checks."""
    pass

def validate_complexity(obj_path: Path, vertices_threshold: int) -> None:
    """
    Validates that the object at obj_path does not exceed vertex and face count thresholds.
    
    :param obj_path: Path to the object file.
    :type obj_path: Path
    :param vertices_threshold: Maximum allowed number of vertices.
    :type vertices_threshold: int
    """
    gltf = GLTF2().load(obj_path)
    
    # Check vertices
    total_vertices = 0
    for mesh in gltf.meshes:
        for primitive in mesh.primitives:
            # Count vertices
            if primitive.attributes.POSITION is not None:
                total_vertices += gltf.accessors[primitive.attributes.POSITION].count
    
    if total_vertices == 0:
        raise ComplexityValidationError(f'[COMPLEXITY CHECK] "{obj_path.stem}": No vertices found')   
    if total_vertices > vertices_threshold:
        raise ComplexityValidationError(
            f'[COMPLEXITY CHECK] "{obj_path.stem}": Too many vertices ({total_vertices} > {vertices_threshold})'
        )
        

# --- IMPORT VALIDATION ---

class ImportValidationError(Exception):
    """Raised when Blender fails to import an object."""
    pass


# --- TEXTURE VALIDATION ---

class TextureValidationError(Exception):
    """Raised when an imported object fails texture checks."""
    pass

def compute_texture_stats(img_array: np.ndarray) -> Tuple[float, float]:
    """
    Compute color variance and entropy for a given RGB image array.
    
    :param img_array: Input image as a NumPy array of shape (H, W, 3) with RGB values in [0, 255].
    :type img_array: np.ndarray
    
    :return: Tuple of (variance, entropy) where variance is a measure of color diversity and entropy measures randomness.
    :rtype: Tuple[float, float]
    """
    # Convert to grayscale using perceptual weights
    gray = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    
    # Variance
    variance = np.var(gray)
    
    # Entropy (histogram-based)
    hist, _ = np.histogram(gray.flatten(), bins = 256, range = (0, 255))
    hist = hist / hist.sum()  # Normalize
    hist = hist[hist > 0]  # Remove zeros
    entropy = calculate_entropy(hist, base = 2)
    
    return variance, entropy

def validate_texture(obj, variance_threshold: float, entropy_threshold: float) -> None:
    """
    Check if the object has materials with valid (non-monochromatic) base color textures.
    
    Specifically checks Base Color/Albedo textures connected to the Principled BSDF shader,
    as these determine the visual appearance. Normal maps, roughness maps, etc. are ignored.
    
    :param obj: Blender object to check.
    :param variance_threshold: Minimum texture color variance.
    :type variance_threshold: float
    :param entropy_threshold: Minimum texture color entropy.
    :type entropy_threshold: float
    """
    if not obj.data.materials:
        raise TextureValidationError(f'[TEXTURE CHECK] "{obj.name}": No materials found')
    
    has_base_color_texture = False
    has_valid_base_color_texture = False
    
    for mat in obj.data.materials:
        if mat and mat.use_nodes:
            # Find Principled BSDF shader node
            bsdf_node = None
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf_node = node
                    break
            
            if not bsdf_node:
                continue
            
            # Check if Base Color input has a texture connected
            base_color_input = bsdf_node.inputs.get('Base Color')
            if not base_color_input or not base_color_input.is_linked:
                continue
            
            # Traverse links to find texture node
            for link in base_color_input.links:
                texture_node = link.from_node
                
                # Handle direct texture connection
                if texture_node.type == 'TEX_IMAGE' and texture_node.image:
                    has_base_color_texture = True
                    
                    # Check packed images (.glb format)
                    if texture_node.image.packed_file:
                        try:
                            pixels = np.array(texture_node.image.pixels[:])
                            pixels = pixels.reshape((texture_node.image.size[1], texture_node.image.size[0], 4))
                            pixels_rgb = (pixels[:, :, :3] * 255).astype(np.uint8)
                            variance, entropy = compute_texture_stats(pixels_rgb)
                            # Check if texture is valid based on variance and entropy thresholds
                            if variance >= variance_threshold and entropy >= entropy_threshold:
                                has_valid_base_color_texture = True
                                break
                        except Exception as e:
                            raise TextureValidationError(f'[TEXTURE CHECK] "{obj.name}": Error checking packed base color texture - {e}')
                    
                    # Fallback: external file textures (.gltf format)
                    elif texture_node.image.filepath:
                        img_path = Path(bpy.path.abspath(texture_node.image.filepath))
                        if img_path.exists():
                            try:
                                img = Image.open(img_path).convert('RGB')
                                img_array = np.array(img)
                                variance, entropy = compute_texture_stats(img_array)
                                # Check if texture is valid based on variance and entropy thresholds    
                                if variance >= variance_threshold and entropy >= entropy_threshold:
                                    has_valid_base_color_texture = True
                                    break
                            except Exception as e:
                                raise TextureValidationError(f'[TEXTURE CHECK] "{obj.name}": Error checking base color texture - {e}')
        
        if has_valid_base_color_texture:
            break
    
    if not has_base_color_texture:
        raise TextureValidationError(f'[TEXTURE CHECK] "{obj.name}": No base color texture found')
    if not has_valid_base_color_texture:
        raise TextureValidationError(f'[TEXTURE CHECK] "{obj.name}": Base color texture is monochromatic or invalid')
    
    
# --- PROTECTED OBJECT VALIDATION ---
class ProtectedObjectValidationError(Exception):
    """Raised when a protected object is not visible in the current view."""
    pass

def validate_protected_objects_visibility(scene_dir: Path, view_idx: int, protected_objects: List[str], object_pass_map: dict) -> None:
    """
    Validate that all protected objects are visible in the specified view.
    
    :param scene_dir: Directory containing the scene data.
    :type scene_dir: Path
    :param view_idx: Index of the view to check.
    :type view_idx: int
    :param protected_objects: List of protected object names to validate.
    :type protected_objects: List[str]
    :param object_pass_map: Mapping from object names to their corresponding pass indices in the object ID maps.
    :type object_pass_map: dict
    """
    # Load object ID maps
    objid = load_exr(f'{scene_dir}/obj_mask_for_view-{view_idx:04d}.exr')
    objid = np.rint(objid).astype(np.int32)
    
    # Check visibility of each protected object
    for protected_obj in protected_objects:
        pass_idx = object_pass_map.get(protected_obj)
        if pass_idx is None:
            raise ProtectedObjectValidationError(f'[PROTECTED OBJECT CHECK] "{protected_obj}": No pass index found in object_pass_map')
        if pass_idx not in objid:
            raise ProtectedObjectValidationError(f'[PROTECTED OBJECT CHECK] "{protected_obj}": Pass index {pass_idx} not found in object ID map for view {view_idx}')
        if np.sum(objid == pass_idx) == 0:
            raise ProtectedObjectValidationError(f'[PROTECTED OBJECT CHECK] "{protected_obj}": Not visible in view {view_idx} (pass index {pass_idx} has zero pixels)')
    
    
# --- OVERLAP VALIDATION ---

def compute_object_surface_overlap(obj_pass_idx_str: str, obj_pass_idx_int: int, depth0: np.ndarray, depth1: np.ndarray, depth0_occ: np.ndarray,
                                   depth1_occ: np.ndarray, objid0: np.ndarray, objid1: np.ndarray, poses0: dict, poses1: dict, camera0: dict, camera1: dict, 
                                   coverage_threshold: float) -> Tuple[bool, float]:
    """
    Compute surface overlap coverage for a specific object between two consecutive views.
    
    The coverage metrics ensure that both views show sufficient common surface:
    - coverage_0_to_1: percentage of view0 pixels that successfully map to view1
    - coverage_1_to_0: percentage of view1 pixels that successfully map to view0
    - min_coverage: minimum of the two coverage values (worst-case overlap)
    
    Parameters:
    - obj_pass_idx_str : str
        Pass index as string (used for transformation lookup in poses dict).
    - obj_pass_idx_int : int
        Pass index as integer (used for mask extraction from objid maps).
    - depth0 : ndarray of shape (H, W)
        Depth map of view0.
    - depth1 : ndarray of shape (H, W)
        Depth map of view1.
    - depth0_occ : ndarray of shape (H, W)
        Minimum depth in 3x3 neighborhood for view0.
    - depth1_occ : ndarray of shape (H, W)
        Minimum depth in 3x3 neighborhood for view1.
    - objid0 : ndarray of shape (H, W)
        Object ID map of view0.
    - objid1 : ndarray of shape (H, W)
        Object ID map of view1.
    - poses0 : dict
        Poses dictionary for view0.
    - poses1 : dict
        Poses dictionary for view1.
    - camera0 : dict
        Camera parameters for view0.
    - camera1 : dict
        Camera parameters for view1.
    - coverage_threshold : float
        Minimum required coverage for both views.
    
    Returns:
    - bool
        True if min_coverage >= threshold, False otherwise.
    - float
        The computed minimum coverage value (worst-case between both directions).
    """
    # Extract masks for the specific object
    mask0 = (objid0 == obj_pass_idx_int)
    mask1 = (objid1 == obj_pass_idx_int)
    
    n_pixels_view0 = np.sum(mask0)
    n_pixels_view1 = np.sum(mask1)
    
    # Handle edge cases
    if n_pixels_view0 == 0 or n_pixels_view1 == 0:
        return False, 0.0
    
    # Get all pixels of the object in view0
    coords0 = np.argwhere(mask0)  # (N, 2) array of (row, col) = (v, u)
    # Map all pixels from view0 to view1
    valid_mask_0_to_1, mapped_coords_0_to_1 = map_points(
        obj_pass_idx_str, coords0, depth0, depth1_occ, poses0, 
        poses1, camera0, camera1, check_occlusion = True
    )
    # For valid mappings, check if they land on the same object in view1
    valid_indices = np.where(valid_mask_0_to_1)[0]
    u_mapped = mapped_coords_0_to_1[valid_indices, 0].astype(np.int32)
    v_mapped = mapped_coords_0_to_1[valid_indices, 1].astype(np.int32)
    # Check if mapped pixels belong to the same object
    lands_on_object = mask1[v_mapped, u_mapped]
    successful_maps_0_to_1 = np.sum(lands_on_object)
    
    # Get all pixels of the object in view1
    coords1 = np.argwhere(mask1)  # (N, 2) array of (row, col) = (v, u)
    # Map all pixels from view1 to view0
    valid_mask_1_to_0, mapped_coords_1_to_0 = map_points(
        obj_pass_idx_str, coords1, depth1, depth0_occ, poses1, 
        poses0, camera1, camera0, check_occlusion = True
    )
    # For valid mappings, check if they land on the same object in view0
    valid_indices = np.where(valid_mask_1_to_0)[0]
    u_mapped = mapped_coords_1_to_0[valid_indices, 0].astype(np.int32)
    v_mapped = mapped_coords_1_to_0[valid_indices, 1].astype(np.int32)
    # Check if mapped pixels belong to the same object
    lands_on_object = mask0[v_mapped, u_mapped]
    successful_maps_1_to_0 = np.sum(lands_on_object)
    
    # Compute coverage metrics (percentage of visible surface that maps successfully)
    coverage_0_to_1 = successful_maps_0_to_1 / n_pixels_view0 if n_pixels_view0 > 0 else 0.0
    coverage_1_to_0 = successful_maps_1_to_0 / n_pixels_view1 if n_pixels_view1 > 0 else 0.0
    
    # Use minimum coverage as the metric
    min_coverage = min(coverage_0_to_1, coverage_1_to_0)
    
    return min_coverage >= coverage_threshold, min_coverage

def validate_view_overlap(scene_dir: str, view0: int, view1: int, coverage_threshold: float, min_common_objects: int, room_indices: List[int] = None) -> Tuple[bool, dict]:
    """
    Validate that common objects between two views have sufficient surface overlap.
    
    Parameters:
    - scene_dir : str or Path
        Path to the scene directory.
    - view0 : int
        First view index.
    - view1 : int
        Second view index.
    - coverage_threshold : float
        Minimum required coverage for each common object.
    - min_common_objects : int
        Minimum number of objects required to have sufficient overlap.
    - room_indices : List[int], optional
        List of pass indices corresponding to room objects to exclude from validation.
    
    Returns:
    - bool
        True if the minimum number of common objects pass the coverage threshold.
    - dict
        Dictionary with pass_index strings as keys and their min_coverage values.
    """
    scene_dir = str(scene_dir)
    
    # Load depth maps
    depth0 = load_exr(f'{scene_dir}/depth-{view0:04d}.exr')
    depth1 = load_exr(f'{scene_dir}/depth-{view1:04d}.exr')
    
    # Compute the minimum depth value in a 3x3 neighborhood around each pixel
    depth0_occ = depth_min3x3(depth0)
    depth1_occ = depth_min3x3(depth1)
    
    # Load object ID maps
    objid0 = load_exr(f'{scene_dir}/obj_mask_for_view-{view0:04d}.exr')
    objid1 = load_exr(f'{scene_dir}/obj_mask_for_view-{view1:04d}.exr')
    
    objid0 = np.rint(objid0).astype(np.int32)
    objid1 = np.rint(objid1).astype(np.int32)
    
    # Load object poses to get pass indices
    with np.load(f'{scene_dir}/objs_per_view_{view0}.npz', allow_pickle = True) as d:
        poses0 = d['poses'].item()
    with np.load(f'{scene_dir}/objs_per_view_{view1}.npz', allow_pickle = True) as d:
        poses1 = d['poses'].item()
    
    # Load camera parameters
    with np.load(f'{scene_dir}/camera{view0}.npz') as d:
        camera0 = {'K': d['K'], 'R': d['R'], 't': d['t']}
    with np.load(f'{scene_dir}/camera{view1}.npz') as d:
        camera1 = {'K': d['K'], 'R': d['R'], 't': d['t']}
    
    # Find common objects - Exclude background (ID 0)
    ids0 = set(np.unique(objid0)) - {0}
    ids1 = set(np.unique(objid1)) - {0}
    common_pass_indices = ids0 & ids1
    
    # Filter out room objects if room_indices is provided
    if room_indices is not None:
        room_indices_set = set(room_indices)
        common_pass_indices = common_pass_indices - room_indices_set
    
    if len(common_pass_indices) == 0:
        # If no common objects, pass if min_common_objects is 0 or less
        return (min_common_objects <= 0), {}
    
    results = {}
    passed = 0
    
    for pass_idx in common_pass_indices:
        # Get integer and string versions of the pass index
        pass_idx_int = int(pass_idx)
        pass_idx_str = str(pass_idx)
        
        # Compute surface overlap for this object
        ok, coverage = compute_object_surface_overlap(
            pass_idx_str, pass_idx_int, depth0, depth1, depth0_occ, 
            depth1_occ, objid0, objid1, poses0, poses1, camera0, camera1,
            coverage_threshold
        )
        
        # Store the coverage result for this object
        results[pass_idx_str] = coverage
        
        if ok:
            passed += 1
            if passed >= min_common_objects:
                # Early exit if we already have enough objects passing the threshold
                return True, results 
    
    return False, results