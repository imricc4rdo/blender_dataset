import gc
import bpy
import sys
import math
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple
BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
from utils.room_utils import Room
from utils.objaverse_utils import *
from utils.validation_utils import *
from mathutils import Euler, Matrix, Vector


def parse_args():
    """
    Parse command-line arguments for the dataset generation script.
    """
    parser = argparse.ArgumentParser(
        description = 'Generate Blender dataset with random scenes'
    )

    # Scene generation settings
    parser.add_argument('--n_scenes', type = int, default = 500,
                        help = 'Number of scenes to generate')

    parser.add_argument('--n_views', type = int, default = 3,
                        help = 'Number of views per scene')
    
    parser.add_argument('--max_view_attempts', type = int, default = 5,
                        help = 'Maximum number of attempts for a view before restarting the scene')
    
    # Object configuration
    parser.add_argument('--min_objects', type = int, default = 3,
                        help = 'Minimum number of objects per view')

    parser.add_argument('--max_objects', type = int, default = 5,
                        help = 'Maximum number of objects per view')

    parser.add_argument('--common_objects', type = int, default = 2,
                        help = 'Minimum number of objects that are common across all views (must be ≤ min_objects)')

    # Room geometry
    parser.add_argument('--area', type = float, default = 2000.0,
                        help = 'Area (X * Y) of the placement area',)
    
    parser.add_argument('--walls_height', type = float, default = 25.0,
                        help = 'Walls height')

    parser.add_argument('--ground_z', type = float, default = 0.0,
                        help = 'Ground Z coordinate')
    
    # Directory paths
    parser.add_argument('--obj_folder', type = Path,
                        default = BASE / 'objects',
                        help = 'Folder containing meshes')
    
    parser.add_argument('--ann_folder', type = Path,
                        default = BASE / 'annotations',
                        help = 'Folder for Objaverse annotations')

    parser.add_argument('--out_folder', type = Path,
                        default = BASE / 'output',
                        help = 'Output folder')

    parser.add_argument('--background_images_folder', type = Path,
                        default = BASE / 'background_images',
                        help = 'Folder containing background images')

    # Validation thresholds
    parser.add_argument('--size_mb_threshold', type = float, default = 40.0,
                        help = 'Maximum file size in MB for objects')
    
    parser.add_argument('--vertices_threshold', type = int, default = 50000,
                        help = 'Maximum number of vertices for objects')
    
    parser.add_argument('--texture_variance_threshold', type = float, default = 450.0,
                        help = 'Minimum texture color variance (rejects monochromatic textures)')
    
    parser.add_argument('--texture_entropy_threshold', type = float, default = 3.5,
                        help = 'Minimum texture entropy (rejects low-entropy textures)')
    
    parser.add_argument('--overlap_coverage_threshold', type = float, default = 0.3,
                        help = 'Minimum coverage threshold for surface overlap between views (0.3 = 30%%)')
    
    # Other parameters
    parser.add_argument('--padding', type = float, default = 0.10,
                        help = 'Extra spacing between objects')
    
    parser.add_argument('--processes', type = int, default = None,
                        help = 'Number of processes to use for downloading objects')

    if '--' in sys.argv:
        # Take only the arguments after '--'
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        # No custom arguments provided → use defaults
        argv = []

    # Parse only the filtered arguments
    return parser.parse_args(argv)


# Customizable parameters
args = parse_args()
# Scene generation settings
N_SCENES = args.n_scenes
N_VIEWS = args.n_views
MAX_VIEW_ATTEMPTS = args.max_view_attempts
# Object configuration
MIN_OBJECTS = args.min_objects
MAX_OBJECTS = args.max_objects
COMMON_OBJECTS = args.common_objects
# Room geometry
AREA = args.area
WALLS_HEIGHT = args.walls_height
GROUND_Z = args.ground_z
# Directory paths
OBJECT_FOLDER = args.obj_folder.resolve()
ANNOTATION_FOLDER = args.ann_folder.resolve()
OUT_FOLDER = args.out_folder.resolve()
BACKGROUND_IMAGES_FOLDER = args.background_images_folder.resolve()
# Validation thresholds
SIZE_MB_THRESHOLD = args.size_mb_threshold
VERTICES_THRESHOLD = args.vertices_threshold
TEXTURE_VARIANCE_THRESHOLD = args.texture_variance_threshold
TEXTURE_ENTROPY_THRESHOLD = args.texture_entropy_threshold
OVERLAP_COVERAGE_THRESHOLD = args.overlap_coverage_threshold
# Other parameters
PADDING = args.padding
PROCESSES = args.processes

ROOM_PLACEMENT_SCALE = 2.0  # Room is larger than placement area by this factor


def ensure_dirs(*paths: Path):
    """
    Remove and recreate directories from scratch.
    
    :param *paths: One or more directory paths to ensure.
    :type *paths: Path
    """
    for p in paths:
        shutil.rmtree(p, ignore_errors = True)
        p.mkdir(parents = True, exist_ok = True)


def enable_render_passes(scene: bpy.types.Scene) -> None:
    """
    Enable all render passes required for exporting (Z-depth pass and Object Index pass).
    
    :param scene: Blender scene.
    :type scene: bpy.types.Scene
    """
    vl = scene.view_layers[0]
    vl.use_pass_z = True                # Depth
    vl.use_pass_object_index = True     # Object ID


def setup_compositor(scene: bpy.types.Scene) -> None:
    """
    Configure the compositor to export two EXR files:
    - One for the Z-depth pass.
    - One for the Object Index pass.
    
    :param scene: Blender scene.
    :type scene: bpy.types.Scene
    """
    scene.use_nodes = True
    tree = scene.node_tree

    # Remove all existing compositor nodes
    for node in list(tree.nodes):
        tree.nodes.remove(node)

    # Create Render Layers node
    rl = tree.nodes.new(type = 'CompositorNodeRLayers')

    # Create File Output node for saving EXR passes
    out = tree.nodes.new(type = 'CompositorNodeOutputFile')
    out.format.file_format = 'OPEN_EXR'
    out.format.color_depth = '32'
    out.format.exr_codec = 'ZIP'

    # Remove automatically created default slots
    out.file_slots.clear()
    # Create file slots for depth + object index
    out.file_slots.new(f'depth')
    out.file_slots.new(f'objid')
    
    # Depth value -> RGBA (replicate on RGB)
    depth_rgba = tree.nodes.new(type='CompositorNodeCombineColor')
    # Replicate the Depth value to R, G, B
    tree.links.new(rl.outputs['Depth'], depth_rgba.inputs['Red'])
    tree.links.new(rl.outputs['Depth'], depth_rgba.inputs['Green'])
    tree.links.new(rl.outputs['Depth'], depth_rgba.inputs['Blue'])
    depth_rgba.inputs['Alpha'].default_value = 1.0
    
    # ObjID value -> RGBA (replicate on RGB)
    objid_rgba = tree.nodes.new(type='CompositorNodeCombineColor')
    # Replicate the IndexOB value to R, G, B
    tree.links.new(rl.outputs['IndexOB'], objid_rgba.inputs['Red'])
    tree.links.new(rl.outputs['IndexOB'], objid_rgba.inputs['Green'])
    tree.links.new(rl.outputs['IndexOB'], objid_rgba.inputs['Blue'])
    objid_rgba.inputs['Alpha'].default_value = 1.0

    # Connect images to file output slots
    tree.links.new(depth_rgba.outputs['Image'], out.inputs[0])  # depth slot
    tree.links.new(objid_rgba.outputs['Image'], out.inputs[1])  # objid slot


def clean_scene(scene: bpy.types.Scene, scene_idx: int) -> None:
    """
    Reset the scene before generating a new one. Behavior:
    - For the first and last call (scene_idx == 0 or scene_idx == N_SCENES), remove all objects.
      This guarantees a fully clean start/end state.
    - Otherwise, remove only mesh objects, keeping cameras and lights.
    
    :param scene: Blender scene.
    :type scene: bpy.types.Scene
    :param scene_idx: Current scene index.
    :type scene_idx: int
    """
    # Reset pass_index counter for new scene
    if 'next_pass_index' in scene:
        del scene['next_pass_index']
    
    # Reset object pass map for new scene
    if 'object_pass_map' in scene:
        del scene['object_pass_map']
    
    # Reset protected objects tracker for new scene
    if 'protected_objects' in scene:
        del scene['protected_objects']
    
    # Reset room indices tracker for new scene
    if 'room_indices' in scene:
        del scene['room_indices']
    
    # Switch from Edit to Object Mode
    if bpy.context.active_object and bpy.context.active_object.mode == 'EDIT':
        bpy.ops.object.mode_set(mode = 'OBJECT')

    # Unhide all objects
    for obj in bpy.data.objects:
        obj.hide_set(False)
        obj.hide_select = False
        obj.hide_viewport = False

    if scene_idx == 0 or scene_idx == N_SCENES:
        # Select and delete all objects in the scene
        bpy.ops.object.select_all(action = 'SELECT')
        bpy.ops.object.delete()

    else:
        # Remove only mesh objects
        to_remove = [obj for obj in scene.objects if obj.type == 'MESH']
        for obj in to_remove:
            bpy.data.objects.remove(obj, do_unlink = True)
            
    # Remove any orphaned data blocks left in memory
    bpy.ops.outliner.orphans_purge(
        do_local_ids = True,
        do_linked_ids = True,
        do_recursive = True
    )
    

def area_bounds(vertices: List[Vector]) -> Tuple[float, float, float, float]:
    """
    Compute AABB bounds in XY plane from a list of vertices.
    Returns: xmin, xmax, ymin, ymax
    
    :param vertices: List of vertices.
    :type vertices: List[Vector]
    
    :return xmin, xmax, ymin, ymax: Bounding box coordinates.
    :rtype xmin, xmax, ymin, ymax: Tuple[float, float, float, float]
    """
    xmin, xmax, ymin, ymax = float('inf'), float('-inf'), float('inf'), float('-inf')
    # Compute bounds
    for v in vertices:
        if v.x < xmin: xmin = v.x
        if v.x > xmax: xmax = v.x
        if v.y < ymin: ymin = v.y
        if v.y > ymax: ymax = v.y
    return xmin, xmax, ymin, ymax


def should_change(p: float = 0.5) -> bool:
    """
    Randomly decides whether to change a scene setting or keep it unchanged.
    
    :param p: Probability of change.
    :type p: float
    
    :return: Whether to change the setting.
    :rtype: bool
    """
    return random.random() > p


def import_object(scene: bpy.types.Scene, obj_path: str) -> bpy.types.Object:
    """
    (Safely) import .glb objects into the scene.
    Merges all meshes into a single mesh, clears parenting, and removes other imported objects.
    """
    def _descendants(obj: bpy.types.Object) -> List[bpy.types.Object]:
        """
        Return all descendants of the given object.
        """
        out = []
        for c in obj.children:
            out.append(c)
            out.extend(_descendants(c))
        return out
    
    def _is_finite_vec3(v: Vector) -> bool:
        """
        Check if all components of a 3D vector are finite numbers.
        """
        return (math.isfinite(float(v[0])) and math.isfinite(float(v[1])) and math.isfinite(float(v[2])))
    
    filepath = str(obj_path)

    # Snapshot of existing objects - before import
    before = set(o.name for o in scene.objects)

    # Import object
    bpy.ops.import_scene.gltf(filepath = filepath)

    # Snapshot of existing objects - after import
    after = set(o.name for o in scene.objects)
    # Find names of newly imported objects
    imported_names = list(after - before)
    # Find references to newly imported objects
    imported_objs = [bpy.data.objects[n] for n in imported_names if n in bpy.data.objects]
    
    # Find all meshes among the imported objects (including those nested under empties)
    meshes = [o for o in imported_objs if o.type == 'MESH']
    empties = [o for o in imported_objs if o.type == 'EMPTY']
    for e in empties:
        meshes.extend([d for d in _descendants(e) if d.type == 'MESH'])
    # Remove duplicates
    meshes = list({m.name: m for m in meshes if m.name in bpy.data.objects}.values())

    if not meshes:
        raise ImportValidationError('No mesh found.')
    
    # Join all meshes (if more than one)
    if len(meshes) > 1:
        bpy.ops.object.select_all(action = 'DESELECT')
        for m in meshes:
            if m.name in bpy.data.objects:
                m.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
    
    # The active object is now the final mesh  
    final_mesh = bpy.context.view_layer.objects.active
    if not final_mesh or final_mesh.type != 'MESH':
        # Fallback: take the last mesh in the list
        remaining = [o for o in scene.objects if o.type == 'MESH' and o.name in bpy.data.objects]
        final_mesh = remaining[-1]
    
    # Clear parenting   
    bpy.ops.object.select_all(action = 'DESELECT')
    final_mesh.select_set(True)
    bpy.context.view_layer.objects.active = final_mesh
    bpy.ops.object.parent_clear(type = 'CLEAR_KEEP_TRANSFORM')
    
    # Center the origin on the geometry (bbox center)
    bpy.ops.object.origin_set(type = 'ORIGIN_GEOMETRY', center = 'BOUNDS')
    
    # Link final mesh to the scene collection (if not already linked)
    if final_mesh.name not in scene.collection.objects:
        scene.collection.objects.link(final_mesh)

    # Unlink final mesh from other collections
    for col in list(final_mesh.users_collection):
        if col != scene.collection:
            col.objects.unlink(final_mesh)

    # Remove other imported objects
    for name in imported_names:
        o = bpy.data.objects.get(name)
        if o and o != final_mesh:
            bpy.data.objects.remove(o, do_unlink = True)

    # Rename final mesh
    final_mesh.name = Path(filepath).stem
    
    # Check for NaN/Inf in vertex coordinates
    for v in final_mesh.data.vertices:
        if not _is_finite_vec3(v.co):
            raise IntegrityValidationError(f'[INTEGRITY CHECK] "{final_mesh.name}" has non-finite vertex coordinates.')
    
    return final_mesh


def is_room_object(obj_name: str) -> bool:
    """
    Check if an object is a room component (floor, ceiling, or wall).
    
    :param obj_name: Name of the object.
    :type obj_name: str
    
    :return: True if object is a room component, False otherwise.
    :rtype: bool
    """
    return obj_name in ['Floor', 'Ceiling'] or obj_name.startswith('Wall_')
    

def assign_object_indices(scene: bpy.types.Scene, room: bool = False) -> None:
    """
    Assign pass_index to meshes, reusing indices for re-imported objects.
    Uses scene['next_pass_index'] counter and scene['object_pass_map'] dict
    to maintain persistent name -> index mapping across views.
    
    :param scene: Blender scene containing the objects.
    :type scene: bpy.types.Scene
    :param room: Whether to track and store pass indices for room objects.
    :type room: bool
    """
    # Init counter and mapping dict if missing
    if 'next_pass_index' not in scene:
        scene['next_pass_index'] = 1
    if 'object_pass_map' not in scene:
        scene['object_pass_map'] = {}
    
    # Get current counter and mapping
    next_idx = int(scene['next_pass_index'])
    pass_map = dict(scene['object_pass_map'])
    
    # Track room object indices
    room_indices = []
    
    # Assign pass_index to meshes
    for obj in scene.objects:
        if obj.type == 'MESH':
            if obj.name in pass_map:
                # Reuse existing pass_index for re-imported objects
                obj.pass_index = pass_map[obj.name]
            else:
                # Assign new pass_index to new objects
                obj.pass_index = next_idx
                pass_map[obj.name] = next_idx
                next_idx += 1
            
            # Track room objects
            if room and is_room_object(obj.name):
                room_indices.append(obj.pass_index)
    
    # Update counter and mapping
    scene['next_pass_index'] = next_idx
    scene['object_pass_map'] = pass_map
    
    # Store room indices in scene properties if requested
    if room:
        scene['room_indices'] = room_indices


def update_scene_objects(scene: bpy.types.Scene, view_idx: int, annotations: dict) -> List[str]:
    """
    Dynamically add or remove objects from the scene across multiple views, while 
    ensuring:
    - At least MIN_OBJECTS remain in the scene
    - At least COMMON_OBJECTS are preserved across all views
    
    Objects that fail import or validation are automatically skipped, and the function
    continues attempting imports until the target count is reached. Successfully imported
    objects are assigned unique pass_index values for rendering.
    
    :param scene: Blender scene to modify.
    :type scene: bpy.types.Scene
    :param view_idx: Current view index (0-based).
    :type view_idx: int
    :param annotations: Objaverse annotations database for object selection.
    :type annotations: dict
    
    :return: List of successfully added object names.
    :rtype: List[str]
    """
    # Current placed mesh objects
    placed_objs = [obj.name for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name)]
    
    # --- ADD ---
    to_import = [] # Initialize list to store names of objects to import
    if (len(placed_objs) < MAX_OBJECTS and should_change()) or view_idx == 0:
        max_add = MAX_OBJECTS - len(placed_objs)
        n_add = random.randint(MIN_OBJECTS, max_add) if view_idx == 0 else random.randint(1, max_add)
        to_import = [f'obj_{i}' for i in range(n_add)] # Placeholder names for objects to import
        # Objects are not actually imported yet, the list is used to track how many objects are intended to be added, 
        # so that importing and immediately removing objects is avoided

    # --- REMOVE (skip for first view) ---
    if view_idx > 0:
        # Get list of protected objects
        protected_objs = list(scene.get('protected_objects', []))
        # Get list of currently placed objects + to_import (which are about to be added)
        all_placed_names = placed_objs + to_import
        
        if len(all_placed_names) > MIN_OBJECTS and should_change():
            # Calculate how many objects can be removed while respecting the minimum object constraint
            max_remove_all = len(all_placed_names) - MIN_OBJECTS
            # List of only non-protected placed objects that can be removed
            removable_placed = [obj for obj in placed_objs if obj not in protected_objs]
            # Calculate how many objects can be removed considering both the minimum object constraint and the protected objects
            max_remove = min(max_remove_all, len(removable_placed) + len(to_import))
            
            if max_remove > 0:
                n_remove = random.randint(1, max_remove)
                # Add to the removable list the objects that are about to be imported
                removable_names = removable_placed + to_import
                # Shuffle the list to ensure random selection of objects to remove
                random.shuffle(removable_names)
                # Select first n_remove objects to remove
                to_remove = removable_names[:n_remove]
                # Execute removals
                for obj_name in to_remove:
                    if obj_name in to_import:
                        # Skip object import if it is in the to_import list
                        to_import = [obj for obj in to_import if obj != obj_name]
                    else:
                        # Remove the object from the scene if already placed
                        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink = True)
                
    # -- IMPORT (if necessary) ---
    added_objs = []
    while len(added_objs) < len(to_import):
        n_add = len(to_import) - len(added_objs)
        to_add_paths = pick_objects(n_add, annotations, placed_objs, OBJECT_FOLDER, PROCESSES, SIZE_MB_THRESHOLD)
        
        for p in to_add_paths:
            # Snapshot objects before import attempt
            before_import = set(obj.name for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name))
            
            try:
                validate_object(p)                                                                  # Check file integrity and format
                validate_complexity(p, VERTICES_THRESHOLD)                                          # Check vertex count limits
                obj = import_object(scene, p)                                                       # Import and merge meshes into a single object
                validate_texture(obj, TEXTURE_VARIANCE_THRESHOLD, TEXTURE_ENTROPY_THRESHOLD)        # Check for valid textures
                added_objs.append(obj.name)
            except (IntegrityValidationError, ComplexityValidationError, ImportValidationError, TextureValidationError) as e:
                # Expected validation failure - message already includes descriptive prefix
                print(f'{e}')
                # Cleanup: remove any partially imported objects
                after_import = set(obj.name for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name))
                partial_imports = after_import - before_import
                for obj_name in partial_imports:
                    if obj_name in bpy.data.objects:
                        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink = True)
                bpy.ops.outliner.orphans_purge(
                    do_local_ids = True,
                    do_linked_ids = True,
                    do_recursive = True
                )
            except Exception as e:
                # Unexpected error during validation/import
                print(f'[UNEXPECTED] {type(e).__name__} for {p.stem}: {e}')
                # Cleanup: remove any partially imported objects
                after_import = set(obj.name for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name))
                partial_imports = after_import - before_import
                for obj_name in partial_imports:
                    if obj_name in bpy.data.objects:
                        bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink = True)
                bpy.ops.outliner.orphans_purge(
                    do_local_ids = True,
                    do_linked_ids = True,
                    do_recursive = True
                )
        
        # Update placed objects list after imports
        placed_objs = [obj.name for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name)]
    
    # Mark protected objects ONLY during view 0
    if view_idx == 0:
        all_objs = [obj.name for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name)]
        # Shuffle for random selection
        random.shuffle(all_objs)
        # First COMMON_OBJECTS are always protected
        protected = all_objs[:COMMON_OBJECTS]
        
        # Remaining objects have 50% chance to be protected
        for obj_name in all_objs[COMMON_OBJECTS:]:
            if should_change():
                protected.append(obj_name)
        
        scene['protected_objects'] = protected
    
    # Assign pass_index values to new objects
    assign_object_indices(scene)
    
    return added_objs


def mesh_bounds_world(obj: bpy.types.Object) -> Tuple[float, float, float, float, float, float]:
    """
    Compute the evaluated mesh AABB (Axis-Aligned Bounding Box) 
    bounds in WORLD space.

    :param obj: Blender object to compute bounds for.
    :type obj: bpy.types.Object
    
    :return: (xmin, xmax, ymin, ymax, zmin, zmax) bounds of the mesh in world space.
    :rtype: Tuple[float, float, float, float, float, float]
    """
    # Get the evaluated dependency graph
    dg = bpy.context.evaluated_depsgraph_get()
    # Get the evaluated version of the object
    oe = obj.evaluated_get(dg)
    # Create a temporary evaluated mesh
    me = oe.to_mesh()
    try:
        # World transformation matrix of the evaluated object
        mw = oe.matrix_world
        xs, ys, zs = [], [], []
        for v in me.vertices:
            w = mw @ v.co
            xs.append(w.x); ys.append(w.y); zs.append(w.z)
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)
    finally:
        # Free the temporary mesh
        oe.to_mesh_clear()
        

def normalize_objects(scene: bpy.types.Scene, added_objs: List[str], volume: float = 380.0) -> None:
    """
    Normalize all new objects to fit within a box of given volume.
    
    :param scene: Blender scene containing the objects.
    :type scene: bpy.types.Scene
    :param added_objs: List of names of newly added objects to normalize.
    :type added_objs: List[str]
    :param volume: Target volume for the bounding box of each object.
    :type volume: float
    """    
    # Process only newly added objects
    for obj_name in added_objs:
        obj = scene.objects[obj_name]
        # Compute object metrics
        xmin, xmax, ymin, ymax, zmin, zmax = mesh_bounds_world(obj)
        xsize, ysize, zsize = xmax - xmin, ymax - ymin, zmax - zmin
        volume_obj = xsize * ysize * zsize
        scale_factor = (volume / volume_obj) ** (1.0 / 3.0)
        scale_factor *= random.uniform(0.9, 1.1)  # Add some randomness to the scaling
        # Apply scaling
        obj.scale = obj.scale * Vector((scale_factor, scale_factor, scale_factor))
        

def restart_scene(scene_idx: int, scene_dir: Path) -> int:
    """
    Restart the current scene by clearing all objects and resetting counters.
    This is used when a view fails to meet validation criteria after multiple attempts.
    
    :param scene_idx: Current scene index.
    :type scene_idx: int
    :param scene_dir: Directory for the current scene, to be cleared.
    :type scene_dir: Path
    
    :return: Updated scene index after restart.
    :rtype: int
    """
    print(f'[SCENE RESTART] Restarting scene #{scene_idx + 1}...')
    scene_idx -= 1
    if scene_dir is not None and scene_dir.exists():
        shutil.rmtree(scene_dir, ignore_errors = True)
    return scene_idx


def ensure_scene_dir(n_objects: int) -> Path:
    """
    Create and return the output directory for the current scene.
    Folder layout: OUT_FOLDER / <n_objects> / <scene_index>.
    
    :param n_objects: Number of objects in the current scene (used for folder organization).
    :type n_objects: int
    
    :return: Path to the created scene directory.
    :rtype: Path
    """
    base_dir = OUT_FOLDER / str(n_objects)
    base_dir.mkdir(parents = True, exist_ok = True)

    # Find the maximum number that already exists
    existing = [int(p.name) for p in base_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    next_index = max(existing, default = -1) + 1

    # Create the new scene directory
    scene_dir = base_dir / f'{next_index:04d}'
    scene_dir.mkdir(parents = True, exist_ok = False)
    return scene_dir
          
            
def random_scaling(scene: bpy.types.Scene, view_idx: int, smin: float = 0.8, smax: float = 1.2) -> None:
    """
    Randomly scale all objects in the scene.
    For view_idx == 0, scaling is always applied.
    For subsequent views, scaling is applied with probability controlled by should_change().
    
    :param scene: Blender scene containing the objects to scale.
    :type scene: bpy.types.Scene
    :param view_idx: Index of the current view.
    :type view_idx: int
    :param smin: Minimum scaling factor.
    :type smin: float
    :param smax: Maximum scaling factor.
    :type smax: float
    """
    # Select objects
    objs = [obj for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name)]
    for obj in objs:
        if view_idx == 0 or should_change():
            s = random.uniform(smin, smax)
            obj.scale = obj.scale * Vector((s, s, s))
            

def store_and_apply_scale(scene: bpy.types.Scene) -> None:
    """
    Store current scale factors in custom properties, then apply scale transforms.
    This preserves scale information before applying transformations to perform
    physics simulation.
    
    :param scene: Blender scene containing the objects to process.
    :type scene: bpy.types.Scene
    """
    objs = [obj for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name)]
    
    for obj in objs:
        # Store or accumulate scale in custom properties
        for axis in ('x', 'y', 'z'):
            key = f'original_scale_{axis}'
            scale_value = getattr(obj.scale, axis)
            obj[key] = obj.get(key, 1.0) * scale_value
        
        # Apply scale transform
        bpy.ops.object.select_all(action = 'DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location = False, rotation = False, scale = True)


def aabb_volume(obj):
    """
    Compute the volume of the object's bounding box.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = mesh_bounds_world(obj)
    xsize, ysize, zsize = xmax - xmin, ymax - ymin, zmax - zmin
    return xsize * ysize * zsize
        
        
def setup_rigid_body(obj: bpy.types.Object, 
                     active: bool = True, 
                     collision_shape: str = 'CONVEX_HULL', 
                     collision_margin: float = 0.001,
                     collision_mesh_source: str = 'FINAL', 
                     friction: float = 0.5, 
                     angular_damping: float = 0.1, 
                     linear_damping: float = 0.04) -> None:
    """
    Set up rigid body physics for the given object with specified parameters.
    
    :param obj: Blender object to configure.
    :type obj: bpy.types.Object
    :param active: Whether the object is active or passive.
    :type active: bool
    :param collision_shape: Shape used for collision detection.
    :type collision_shape: str
    :param collision_margin: Margin for collision detection.
    :type collision_margin: float
    :param collision_mesh_source: Source of the collision mesh.
    :type collision_mesh_source: str
    :param friction: Friction coefficient.
    :type friction: float
    :param angular_damping: Angular damping factor.
    :type angular_damping: float
    :param linear_damping: Linear damping factor.
    :type linear_damping: float
    """
    # Configure rigid body physics for the given object with specified parameters.
    # (Blender defaults)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.rigidbody.object_add()
    obj.rigid_body.type = 'ACTIVE' if active else 'PASSIVE'
    obj.rigid_body.collision_shape = collision_shape
    obj.rigid_body.collision_margin = collision_margin
    obj.rigid_body.use_margin = True
    obj.rigid_body.mesh_source = collision_mesh_source
    obj.rigid_body.friction = friction
    obj.rigid_body.angular_damping = angular_damping
    obj.rigid_body.linear_damping = linear_damping
    obj.rigid_body.mass = aabb_volume(obj)  # Mass is set equal to the volume of the bounding box
        

def initialize_object(obj: bpy.types.Object, num_active_objs: int, 
                      placement_bounds: Tuple[float, float, float, float]) -> None:
    """
    Initialize object position and rotation before physics simulation.
    Places objects at increasing heights and randomly rotates them.
    
    :param obj: Blender object to initialize.
    :type obj: bpy.types.Object
    :param num_active_objs: Number of active objects already placed (used to determine height).
    :type num_active_objs: int
    :param placement_bounds: Tuple containing room bounds (xmin, xmax, ymin, ymax) for initial object placement.
    :type placement_bounds: Tuple[float, float, float, float]
    """
    # Get placement area bounds
    xmin, xmax, ymin, ymax = placement_bounds
    
    # Restrict area for safer placement
    xmin, xmax, ymin, ymax = 0.7 * xmin, 0.7 * xmax, 0.7 * ymin, 0.7 * ymax    
    
    # Set random position
    obj.location.x = random.uniform(xmin, xmax)
    obj.location.y = random.uniform(ymin, ymax)
    obj.location.z = num_active_objs * 15  # Place each object at increasing heights
    
    # Set random rotation
    obj.rotation_mode = 'XYZ'
    obj.rotation_euler = Euler((random.uniform(0, 2 * math.pi),
                                random.uniform(0, 2 * math.pi),
                                random.uniform(0, 2 * math.pi)), 'XYZ')
    

def seconds_to_frames(scene: bpy.types.Scene, seconds: float) -> int:
    """
    Convert seconds to frames based on the scene's frame rate.
    
    :param scene: Blender scene containing the objects to simulate.
    :type scene: bpy.types.Scene
    :param seconds: Time in seconds to convert to frames.
    :type seconds: float
    
    :return: Equivalent time in frames.
    :rtype: int
    """
    return int(seconds * scene.render.fps)


def rotation_delta(old_rot_v: Tuple[float, float, float], new_rot_v: Tuple[float, float, float]) -> float:
    """
    Compute the rotation difference between two Euler rotations.
    
    :param old_rot_v: Old rotation as a tuple of Euler angles (in radians).
    :type old_rot_v: Tuple[float, float, float]
    :param new_rot_v: New rotation as a tuple of Euler angles (in radians).
    :type new_rot_v: Tuple[float, float, float]
    
    :return: Rotation difference in radians.
    :rtype: float
    """
    e_old = Euler(old_rot_v, 'XYZ')
    q_old = e_old.to_quaternion()
    e_new = Euler(new_rot_v, 'XYZ')
    q_new = e_new.to_quaternion()
    return q_new.rotation_difference(q_old).angle
        
        
def objects_not_moving(old_states: dict, new_states: dict, object_stopped_location_threshold: float, 
                       object_stopped_rotation_threshold: float) -> bool:
    """
    Check if all active rigid body objects in the scene have linear and angular velocities below the threshold.
    
    :param old_states: Dictionary containing the previous states of the objects (location and rotation).
    :type old_states: dict
    :param new_states: Dictionary containing the current states of the objects (location and rotation).
    :type new_states: dict
    :param object_stopped_location_threshold: Threshold for determining if an object has stopped moving based on location change (in Blender units).
    :type object_stopped_location_threshold: float
    :param object_stopped_rotation_threshold: Threshold for determining if an object has stopped moving based on rotation change (in radians).
    :type object_stopped_rotation_threshold: float
    
    :return: True if all objects have stopped moving, False otherwise.
    :rtype: bool
    """
    for obj in new_states.keys():
        old_loc, old_rot = old_states[obj]              # Get old location and rotation
        new_loc, new_rot = new_states[obj]              # Get new location and rotation
        
        loc_diff = (new_loc - old_loc).length           # Compute location difference  
        rot_diff = rotation_delta(old_rot, new_rot)     # Compute rotation difference
        
        if loc_diff > object_stopped_location_threshold or rot_diff > object_stopped_rotation_threshold:
            return False
        
    return True


def run_simulation(scene: bpy.types.Scene, min_simulation_time: float = 10.0, max_simulation_time: float = 40.0, 
                   check_object_interval: float = 6.0, object_stopped_location_threshold: float = 0.01, 
                   object_stopped_rotation_threshold: float = 0.05) -> None:
    """
    Run the physics simulation for the specified number of frames.
    The simulation runs in increments of check_object_interval seconds, checking after each increment 
    if all objects have stopped moving. In that case, the simulation ends early to save time.
    
    :param scene: Blender scene containing the objects to simulate.
    :type scene: bpy.types.Scene
    :param min_simulation_time: Minimum simulation time in seconds before checking for object movement.
    :type min_simulation_time: float
    :param max_simulation_time: Maximum simulation time in seconds to run if objects do not stop moving.
    :type max_simulation_time: float
    :param check_object_interval: Time interval in seconds to check if objects have stopped moving.
    :type check_object_interval: float
    :param object_stopped_location_threshold: Threshold for determining if an object has stopped moving based on location change (in Blender units).
    :type object_stopped_location_threshold: float
    :param object_stopped_rotation_threshold: Threshold for determining if an object has stopped moving based on rotation change (in radians).
    :type object_stopped_rotation_threshold: float
    """
    # Clear existing bakes
    bpy.ops.ptcache.free_bake_all()
    
    # Configure simulator
    bpy.context.scene.rigidbody_world.substeps_per_frame = 10
    bpy.context.scene.rigidbody_world.solver_iterations = 10
    bpy.context.scene.rigidbody_world.enabled = True
    
    # Setup point cache
    point_cache = bpy.context.scene.rigidbody_world.point_cache
    point_cache.frame_start = 1
    
    # Run simulation starting from min to max in the configured steps
    for current_time in np.arange(min_simulation_time, max_simulation_time, check_object_interval):
        # Compute current frame
        current_frame = seconds_to_frames(scene, current_time)
        
        # Bake the simulation up to current_frame -> Calculate physics from frame 1 to current_frame
        point_cache.frame_end = current_frame
        with bpy.context.temp_override(point_cache = point_cache):
            bpy.ops.ptcache.bake(bake = True)
        
        # Get object states at current_frame - 1 and current_frame
        scene.frame_set(current_frame - seconds_to_frames(scene, 1))
        old_states = {
            obj: (bpy.context.scene.objects[obj.name].matrix_world.translation.copy(), 
                  Vector(bpy.context.scene.objects[obj.name].matrix_world.to_euler())) 
            for obj in scene.objects 
            if obj.rigid_body and obj.rigid_body.type == 'ACTIVE'
        }
        scene.frame_set(current_frame)
        new_states = {
            obj: (bpy.context.scene.objects[obj.name].matrix_world.translation.copy(), 
                  Vector(bpy.context.scene.objects[obj.name].matrix_world.to_euler())) 
            for obj in scene.objects 
            if obj.rigid_body and obj.rigid_body.type == 'ACTIVE'
        }
        
        # Check if objects have stopped moving
        if objects_not_moving(old_states, new_states, object_stopped_location_threshold, object_stopped_rotation_threshold):
            break
        
        # Free the cache to prepare for the next simulation iteration
        if current_time + check_object_interval < max_simulation_time:
            with bpy.context.temp_override(point_cache = point_cache):
                bpy.ops.ptcache.free_bake()
              
    # Refresh the view layer to ensure final positions are updated
    bpy.context.view_layer.update()


def place_objects(scene: bpy.types.Scene, placement_bounds: Tuple[float, float, float, float]) -> None:
    """
    Arrange objects in the scene using physics simulation. 
    Objects are dropped from stacked positions and allowed to fall and settle naturally.
    Note: Room ceiling object is excluded from physics simulation entirely.
    
    :param scene: Blender scene containing the objects to arrange.
    :type scene: bpy.types.Scene
    :param placement_bounds: Tuple containing room bounds (xmin, xmax, ymin, ymax) for initial object placement.
    :type placement_bounds: Tuple[float, float, float, float]
    """
    # Store scale factors and apply scale transforms before physics
    # This ensures accurate physics collision detection while preserving scale data
    store_and_apply_scale(scene)
    
    # Select all objects in the scene
    objs = [obj for obj in scene.objects if obj.type == 'MESH']
    # Shuffle objects
    random.shuffle(objs)

    # List of active rigid body objects
    active_objs = []
    # Set to frame 1 before adding physics (simulation start frame)
    scene.frame_set(1) 
    # Setup rigid bodies
    for obj in objs:
        
        # Skip ceiling object - it should not interact with other objects
        if obj.name == 'Ceiling':
            continue
        
        # Room objects - set as passive rigid bodies
        if is_room_object(obj.name):
            setup_rigid_body(obj, active = False, collision_shape = 'MESH')
            continue
        
        # Any other object - set as active rigid body
        setup_rigid_body(obj)
        # Add to active objects list
        active_objs.append(obj)
        # Set random initial position
        initialize_object(obj, len(active_objs), placement_bounds)
    
    # Run the simulation
    run_simulation(scene)
    
    # Remove rigid bodies while preserving final simulated positions
    for obj in active_objs:
        
        # Save current world transformation matrix
        mw = obj.matrix_world.copy()
        
        # Remove rigid body
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_remove()
        
        # Restore saved world transformation matrix
        obj.matrix_world = mw


def validate_placement(scene: bpy.types.Scene, room_bounds: Tuple[float, float, float, float], tolerance: float = 0.01) -> bool:
    """
    Validate that all objects are placed within the room bounds.
    
    :param scene: Blender scene.
    :type scene: bpy.types.Scene
    :param room_bounds: Tuple containing room bounds (xmin, xmax, ymin, ymax).
    :type room_bounds: Tuple[float, float, float, float]
    :param tolerance: Allowed tolerance for boundary violations (in Blender units).
    :type tolerance: float
    
    :return: True if all objects are within bounds, False otherwise.
    :rtype: bool
    """
    xmin_r, xmax_r, ymin_r, ymax_r = room_bounds
    
    # Select all objects in the scene (excluding room objects)
    objs = [
        obj for obj in scene.objects 
        if obj.type == 'MESH' and not is_room_object(obj.name)
    ]
    
    for obj in objs:
        xmin_o, xmax_o, ymin_o, ymax_o, zmin_o, zmax_o = mesh_bounds_world(obj)
        
        # Check if object exceeds room bounds (with tolerance)
        if (xmin_o < xmin_r - tolerance or xmax_o > xmax_r + tolerance or
            ymin_o < ymin_r - tolerance or ymax_o > ymax_r + tolerance or
            zmin_o < GROUND_Z - tolerance or zmax_o > WALLS_HEIGHT + tolerance):
            return False
    
    return True


def capture_mesh_states(scene: bpy.types.Scene) -> dict:
    """
    Capture per-object pose parameters for all mesh objects in the scene.
    Returns a dictionary keyed by object name. Each entry contains:
    - 'rot_euler': (rx, ry, rz) 
    - 't'        : (tx, ty, tz) 
    - 'scale'    : (sx, sy, sz)
    
    :param scene: Blender scene containing the objects to capture.
    :type scene: bpy.types.Scene
    
    :return: Dictionary of object states.
    :rtype: dict
    """
    states = {}
    for obj in scene.objects:
        # Iterate all objects (meshes) in scene
        if obj.type == 'MESH':
            states[str(obj.pass_index)] = {
                'rot_euler': (float(obj.rotation_euler.x),                  # Euler rotation (XYZ) in radians
                              float(obj.rotation_euler.y),
                              float(obj.rotation_euler.z)),
                't': (float(obj.location.x),                                # Location (x, y, z) in world space
                      float(obj.location.y), 
                      float(obj.location.z)),
                'scale': (float(obj.get('original_scale_x', 1.0)),          # Per-axis scale
                          float(obj.get('original_scale_y', 1.0)), 
                          float(obj.get('original_scale_z', 1.0)))
            }
    return states


def group_aabb_metrics(objs):
    """
    Return group AABB (containing all the objects) info in world space:
    - center: Vector((cx, cy, cz))
    - radius_xy: 0.5 * sqrt(x * x + y * y)
    - zsize: max(Z) - min(Z)
    - zmax: max(Z)
    """
    # Compute overall AABB
    xmin, ymin, zmin = float('inf'), float('inf'), float('inf')
    xmax, ymax, zmax = float('-inf'), float('-inf'), float('-inf')
    for obj in objs:
        xmin_o, xmax_o, ymin_o, ymax_o, zmin_o, zmax_o = mesh_bounds_world(obj)
        xmin = min(xmin, xmin_o); xmax = max(xmax, xmax_o)
        ymin = min(ymin, ymin_o); ymax = max(ymax, ymax_o)
        zmin = min(zmin, zmin_o); zmax = max(zmax, zmax_o)
    
    # Compute the center of the AABB as the average between min and max values for each axis
    center = Vector(((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2))
    
    # Overall extents of the group
    xsize = xmax - xmin; ysize = ymax - ymin; zsize = zmax - zmin
    
    # Radius in the XY plane
    radius_xy = 0.5 * math.sqrt(xsize * xsize + ysize * ysize)
    
    return center, radius_xy, zsize, zmax


def setup_light(scene: bpy.types.Scene, placement_bounds: Tuple[float, float, float, float]) -> None:
    """
    Place a light at a random position near the objects with random intensity.
    
    :param scene: Blender scene to modify.
    :type scene: bpy.types.Scene
    :param placement_bounds: Tuple containing room bounds (xmin, xmax, ymin, ymax) for light placement.
    :type placement_bounds: Tuple[float, float, float, float]
    """
    # Create or find existing light
    light_obj = bpy.data.objects.get('Light')
    if light_obj is None:
        light_data = bpy.data.lights.new(name = 'Light', type = 'POINT')
        light_obj = bpy.data.objects.new(name = 'Light', object_data = light_data)
        scene.collection.objects.link(light_obj)

    # Select objects in the scene
    objs = [obj for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name)]
    # Compute metrics of the group
    target, _, height, zmax = group_aabb_metrics(objs)

    # Set (random) location
    xmin, xmax, ymin, ymax = placement_bounds
    pos_x = random.uniform(xmin, xmax)
    pos_y = random.uniform(ymin, ymax)
    pos_z = np.clip(zmax + height * random.uniform(1.50, 2.50), 0.6 * WALLS_HEIGHT, WALLS_HEIGHT - 1.0)
    light_obj.location = (pos_x, pos_y, pos_z)

    # Set (random) intensity
    dist = (light_obj.location - target).length  # Distance from the center of the AABB
    k = random.uniform(35.0, 55.0)
    # Energy scales with dist^2
    light_obj.data.energy = k * (dist * dist)
    
    
def setup_camera(scene: bpy.types.Scene, placement_bounds: Tuple[float, float, float, float], view_attempt: int) -> None:
    """
    Place the camera at a random position and make it look at the center of the room (0, 0).
    
    Camera positioning strategy:
    - Attempt 0: Random initial azimuth around placement area
    - Attempts 1+: Rotate 72° * attempt around placement area from base azimuth
    
    :param scene: Blender scene.
    :type scene: bpy.types.Scene
    :param placement_bounds: Tuple (xmin, xmax, ymin, ymax) of placement area.
    :type placement_bounds: Tuple[float, float, float, float]
    :param view_attempt: Current attempt number for this view.
    :type view_attempt: int
    """
    # Create or find existing camera
    cam_obj = scene.camera
    if cam_obj is None:
        cam_data = bpy.data.cameras.new('Camera')
        cam_obj = bpy.data.objects.new('Camera', cam_data)
        scene.collection.objects.link(cam_obj)
        scene.camera = cam_obj
        # Optical settings
        cam_obj.data.lens_unit = 'MILLIMETERS'
        cam_obj.data.lens = 25
        cam_obj.data.sensor_width = 30      
        cam_obj.data.sensor_height = 30     
        
    # Get placement area bounds and compute radius
    xmin_p, xmax_p, ymin_p, ymax_p = placement_bounds
    placement_radius = 0.5 * math.sqrt((xmax_p - xmin_p) ** 2 + (ymax_p - ymin_p) ** 2)
    
    # Max distance factor to ensure camera is placed within the room but outside the placement area
    max_distance_factor = ROOM_PLACEMENT_SCALE / math.sqrt(2)
    max_distance = placement_radius * max_distance_factor - 0.1  # Keep some margin from the walls
    
    # Distance from the center of the room
    distance = random.uniform(placement_radius + 0.1, max_distance)     
    
    # Store base azimuth
    if view_attempt == 0:
        scene['camera_base_azimuth'] = random.uniform(0.0, 360.0)
    
    # Retrieve base azimuth value
    base_azimuth = scene['camera_base_azimuth']
    
    # Apply systematic rotation for retries (72° spacing for 5 attempts)
    azimuth_degrees = (base_azimuth + 72.0 * view_attempt) % 360.0
    azimuth = math.radians(azimuth_degrees)
    
    # Convert spherical to Cartesian coordinates (XY plane)
    pos_x = distance * math.cos(azimuth)
    pos_y = distance * math.sin(azimuth)
    
    # Set Z position independently (vertical height within walls)
    pos_z = random.uniform(0.7 * WALLS_HEIGHT, 0.9 * WALLS_HEIGHT)
    
    # cam_obj.location = Vector((pos_x, pos_y, pos_z))
    cam_obj.location = (pos_x, pos_y, pos_z)

    # Set target to center of the room (0, 0, 0)
    target = Vector((0.0, 0.0, 0.0))
    # Make the camera look at the room center
    direction = (target - cam_obj.location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    

def get_render_resolution(scene: bpy.types.Scene) -> Tuple[int, int]:
    """
    Compute the effective render resolution in pixels. Used to derive intrinsics in pixels.
    
    :param scene: Blender scene containing the render settings.
    :type scene: bpy.types.Scene
    
    :return: Tuple (width, height) of the effective render resolution in pixels.
    :rtype: Tuple[int, int]
    """
    scale = scene.render.resolution_percentage / 100.0                  # Render scale
    width = int(scene.render.resolution_x * scale)                      # Effective width  in px
    height = int(scene.render.resolution_y * scale)                     # Effective height in px
    return width, height


def compute_intrinsics(cam_obj: bpy.types.Object, scene: bpy.types.Scene) -> np.ndarray:
    """
    Convert Blender camera parameters into the intrinsic matrix K in pixels (OpenCV convention).
    
    :param cam_obj: Blender camera object containing the camera parameters.
    :type cam_obj: bpy.types.Object
    :param scene: Blender scene containing the render settings (used to compute effective resolution).
    :type scene: bpy.types.Scene
    
    :return: Intrinsic matrix K in pixels (OpenCV convention).
    :rtype: np.ndarray 
    """
    cam_data = cam_obj.data                                                 # Get the camera
    w, h = get_render_resolution(scene)                                     # Effective render resolution (px)
    px_aspect = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y   # Pixel aspect ratio

    f_mm = cam_data.lens                 # Focal length in mm
    sw = cam_data.sensor_width           # Sensor width in mm
    sh = cam_data.sensor_height          # Sensor height in mm

    # Derive pixels-per-mm along X/Y, honoring sensor_fit and pixel aspect
    if cam_data.sensor_fit == 'VERTICAL':
        s_u = (w / sw) / px_aspect
        s_v = (h / sh)
    else: # 'HORIZONTAL' or 'AUTO'
        s_u = (w / sw)
        s_v = (h / sh) * px_aspect
    fx = f_mm * s_u
    fy = f_mm * s_v

    cx = w * (0.5 - cam_data.shift_x)            # Principal point x (center of image)
    cy = h * (0.5 + cam_data.shift_y)            # Principal point y (center of image)

    # Intrinsic matrix
    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype = np.float64)
    return K


def compute_extrinsics(cam_obj: bpy.types.Object) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract [R|t] that maps world points to camera coordinates (OpenCV convention).
    
    :param cam_obj: Blender camera object containing the camera parameters.
    :type cam_obj: bpy.types.Object
    
    :return: Tuple (R, t) where R is the rotation matrix and t is the translation vector.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    # Estrai rotazione pura e posizione
    loc, rot, _ = cam_obj.matrix_world.decompose()
    R_wc = rot.to_matrix()
    R_cw = R_wc.transposed()
    t_cw = -(R_cw @ loc)

    # BlenderCam -> OpenCVCam
    R_bcam2cv = Matrix(((1, 0, 0),
                        (0,-1, 0),
                        (0, 0,-1)))
    R = R_bcam2cv @ R_cw
    t = R_bcam2cv @ t_cw

    R = np.array(R, dtype = np.float64)
    t = np.array([t.x, t.y, t.z], dtype = np.float64)
    return R, t


def save_camera_npz(scene: bpy.types.Scene, scene_dir: str, view_idx: int) -> None:
    """
    Save camera intrinsics (K) and extrinsics (R, t) to a .npz file.
    
    :param scene: Blender scene containing the camera.
    :type scene: bpy.types.Scene
    :param scene_dir: Directory where the .npz file will be saved.
    :type scene_dir: str
    :param view_idx: Index of the current view (used for naming the .npz file).
    :type view_idx: int
    """
    cam_obj = scene.camera
    K = compute_intrinsics(cam_obj, scene)      # Compute intrinsics matrix K
    R, t = compute_extrinsics(cam_obj)          # Compute extrinsics R, t
    np.savez(str(scene_dir / f'camera{view_idx}.npz'), K = K, R = R, t = t)


def configure_compositor_outputs(scene: bpy.types.Scene, scene_dir: str, view_idx: int) -> None:
    """
    Update the compositor File Output node for the current view.
    This sets the output directory and filename prefixes for the depth and object mask EXR files.
    
    :param scene: Blender scene containing the compositor to configure.
    :type scene: bpy.types.Scene
    :param scene_dir: Directory where the EXR files will be saved.
    :type scene_dir: str
    :param view_idx: Index of the current view (used for naming outputs).
    :type view_idx: int
    """
    tree = scene.node_tree

    # Retrieve the File Output node
    out = tree.nodes['File Output']

    # Set the scene frame to the current view index
    # (this determines the #### suffix in the output filenames)
    scene.frame_set(view_idx)

    # Set the folder where EXR files will be saved
    out.base_path = str(scene_dir)

    # Set filename prefixes for each output slot
    # Blender will append the frame number (####) automatically
    out.file_slots[0].path = f'depth-'
    out.file_slots[1].path = f'obj_mask_for_view-'


def take_picture(scene: bpy.types.Scene, scene_dir: str, view_idx: int, validation: bool = False):
    """
    Render RGB image (PNG) + Depth pass (EXR) + Object Index pass (EXR) for the current scene configuration.
    RGB image is saved only if validation == False, otherwise only EXR files are generated for validation purposes.
    
    :param scene: Blender scene to render.
    :type scene: bpy.types.Scene
    :param scene_dir: Directory where the rendered files will be saved.
    :type scene_dir: str
    :param view_idx: Index of the current view (used for naming outputs).
    :type view_idx: int
    :param validation: If True, only EXR files are generated for validation purposes. If False, RGB image is also rendered and saved.
    :type validation: bool
    """
    # Rename compositing outputs
    configure_compositor_outputs(scene, scene_dir, view_idx)
    
    if validation:
        # During validation, only depth and object mask EXR files are needed
        # RGB image saving is skipped to save time and disk space
        old_samples = scene.cycles.samples  # Store original sample count
        scene.cycles.samples = 1  # Reduce samples for faster rendering during validation        
        bpy.ops.render.render(write_still = False)  # Skip saving the RGB image
        scene.cycles.samples = old_samples  # Restore original sample count
        
    else:
        # Once the view has been validated, save the RGB image
        scene.render.filepath = str(scene_dir / f'render{view_idx}.png')  # Output RGB filename
        bpy.ops.render.render(write_still = True)  # Render and save the RGB image

# -----------------------------------------------------------------------------------

# Empty the folders from previous runs / ensure they exist
ensure_dirs(OBJECT_FOLDER, OUT_FOLDER)

# Get the current active scene
scene = bpy.context.scene

# Rendering parameters
scene.render.engine = 'CYCLES'
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'

# Rendering parameters tuned for high-speed rendering
# RGB may be noisier, while depth and object-index passes remain deterministic
scene.cycles.samples = 64                       # Same as BLENDER_EEVEE_NEXT (Default = 4096)
scene.render.use_simplify = True                # Default = False
scene.render.simplify_subdivision_render = 0
scene.cycles.max_bounces = 6                    # Default = 12
scene.cycles.diffuse_bounces = 2                # Default = 4
scene.cycles.glossy_bounces = 2                 # Default = 4
scene.cycles.transmission_bounces = 6           # Default = 12
scene.cycles.transparent_max_bounces = 4        # Default = 8
scene.cycles.denoiser = 'OPTIX'                 # Default = 'OPENIMAGEDENOISE'
scene.render.use_persistent_data = True         # Default = False
scene.cycles.adaptive_threshold = 0.02          # Default = 0.01
scene.cycles.adaptive_min_samples = 8           # Default = 0 

# ---
scene.cycles.device = 'GPU'
prefs = bpy.context.preferences
cprefs = prefs.addons['cycles'].preferences
cprefs.get_devices()  # Get the list of compute devices
available_types = {d.type for d in cprefs.devices}  # Available device types
# Select device type -> OPTIX > CUDA > NONE (CPU)
if 'OPTIX' in available_types:
    cprefs.compute_device_type = 'OPTIX'
elif 'CUDA' in available_types:
    cprefs.compute_device_type = 'CUDA'
else:
    cprefs.compute_device_type = 'NONE'  # Fallback to CPU
cprefs.get_devices()  # Update devices
for d in cprefs.devices:
    d.use = (d.type == cprefs.compute_device_type)
# Print selected device info
print(f'[INFO] Compute device type: {cprefs.compute_device_type}')
print(f'[INFO] Enabled devices: {[d.name for d in cprefs.devices if d.use]}')
# ---

# Enable Z-pass and Object-ID pass
enable_render_passes(scene)

# Configure the compositor
setup_compositor(scene)

# Download / Load annotations
print('\n[INFO] Loading annotations...')
annotations = get_annotations(ANNOTATION_FOLDER)

    
print('\n[INFO] Starting scenes generation...')
    
# Generate scenes until N_SCENES successfully completed 
scene_idx = 0
while scene_idx < N_SCENES: 

    print(f'\n[INFO] Scene #{scene_idx + 1}')

    # Clean the scene
    clean_scene(scene, scene_idx)
        
    # Initialize room and build it
    room = Room(
        AREA,
        ROOM_PLACEMENT_SCALE,
        WALLS_HEIGHT,
        BACKGROUND_IMAGES_FOLDER
    )
    room.build()
        
    # Assign pass_index values to room objects and store indices in scene properties
    assign_object_indices(scene, room = True)

    # Initialize scene_dir and error flag before the view loop
    scene_dir = None
    error = False

    # Generate views for the current scene
    for view_idx in range(N_VIEWS):
                        
        # Attempt loop for current view (camera/light repositioning)
        view_attempt = 0
        view_valid = False
        while view_attempt < MAX_VIEW_ATTEMPTS and not view_valid:

            # Only for the first attempt of the each view:
            # - Show placement walls and compute their bounds
            # - Update objects in the scene and normalize new ones
            # - Apply random scaling to all objects
            # - Place objects using physics simulation
            # - Show room walls and compute their bounds
            # - Validate object placement using room bounds
            # - Set lighting
            if view_attempt == 0:
                    
                # Compute placement area vertices and bounds
                placement_vertices = room.show_placement_walls()
                placement_bounds = area_bounds(placement_vertices)
                    
                try:
                    added_objs = update_scene_objects(scene, view_idx, annotations)
                    if added_objs:  # Normalize new objects (if any)
                        normalize_objects(scene, added_objs)
                    n_objs = len([obj for obj in scene.objects if obj.type == 'MESH' and not is_room_object(obj.name)])
                    error = False
                except Exception as e:
                    print(f'[UNEXPECTED] Error while adding/removing objects. {type(e).__name__}: {e}')
                    scene_idx = restart_scene(scene_idx, scene_dir)
                    error = True
                    break
                    
                # If it is the first view of the scene, create the output folder
                if view_idx == 0:
                    n_objs = len(added_objs)
                    scene_dir = ensure_scene_dir(n_objs)
                    
                # Apply random scaling
                random_scaling(scene, view_idx)
                    
                # Place objects
                place_objects(scene, placement_bounds)
                    
                # Compute room vertices and bounds
                room_vertices = room.show_room_walls()
                room_bounds = area_bounds(room_vertices)
                    
                # Validate object placement using room bounds
                if not validate_placement(scene, room_bounds):
                    print(f'[PLACEMENT] Invalid object placement detected.')
                    scene_idx = restart_scene(scene_idx, scene_dir)
                    error = True
                    break
                    
                # Save object data of current picture
                states = capture_mesh_states(scene)
                out = scene_dir / f'objs_per_view_{view_idx}.npz'
                np.savez(str(out), poses = states)
                    
                # Set lighting
                setup_light(scene, placement_bounds)
                
            # Set camera - This is done in every attempt
            setup_camera(scene, placement_bounds, view_attempt)
                
            # Take picture (validation render for view 1+, full render for view 0)
            if view_idx == 0:
                # For the first view, RGB image can be saved immediately since overlap is not validated
                take_picture(scene, scene_dir, view_idx)
            else:
                # For following views, skip saving the RGB image until overlap is validated
                take_picture(scene, scene_dir, view_idx, validation = True)
                    
            # Save camera settings
            save_camera_npz(scene, scene_dir, view_idx)
                
            # Check if protected objects are visible in the current view
            try:
                validate_protected_objects_visibility(scene_dir, view_idx, scene['protected_objects'], scene['object_pass_map'])
            except Exception as e:
                print(f'{e}')
                view_attempt += 1
                    
                if view_attempt >= MAX_VIEW_ATTEMPTS:
                    print(f'[WARN] Max retries reached for view {view_idx}.')
                    scene_idx = restart_scene(scene_idx, scene_dir)
                    error = True
                    break
                    
                continue  # Retry from camera setup
                
            # Starting from view1, validate surface overlap with previous view
            if view_idx > 0:
                try:
                    valid, coverage_results = validate_view_overlap(
                        scene_dir, view_idx - 1, view_idx, OVERLAP_COVERAGE_THRESHOLD, COMMON_OBJECTS, scene['room_indices']
                    )
                       
                    if valid:
                        view_valid = True
                            
                        # Save RGB image after validation 
                        take_picture(scene, scene_dir, view_idx)
                            
                    else:
                        failed_objs = [obj for obj, cov in coverage_results.items() if cov < OVERLAP_COVERAGE_THRESHOLD]
                        print(f'[OVERLAP] {len(failed_objs)} object(s) failed: {failed_objs}')
                        print(f'[OVERLAP] Coverage values: {coverage_results}')
                        view_attempt += 1
                            
                        if view_attempt >= MAX_VIEW_ATTEMPTS:
                            print(f'[WARN] Max retries reached for view {view_idx}.')
                            scene_idx = restart_scene(scene_idx, scene_dir)
                            error = True
                            break
                            
                        continue  # Retry from camera setup
                            
                except Exception as e:
                    print(f'[UNEXPECTED] Error during overlap validation. {type(e).__name__}: {e}')
                    scene_idx = restart_scene(scene_idx, scene_dir)
                    error = True
                    break
                
            # View 0 always valid (no previous view to compare)
            else:
                view_valid = True               
            
        # Clean up after all view attempts (successful or failed)
        gc.collect()
            
        # If this view failed after all retries, break from view loop
        if not view_valid or error:
            break

    # Empty OBJECT_FOLDER to avoid accumulation of downloaded files
    shutil.rmtree(OBJECT_FOLDER, ignore_errors = True)
    OBJECT_FOLDER.mkdir(parents = True, exist_ok = True)

    # Increment scene index
    scene_idx += 1

# Final cleaning
clean_scene(scene, scene_idx)