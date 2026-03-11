import bpy
import bmesh
import random
import numpy as np
from typing import List
from pathlib import Path
from mathutils import Vector


class Room():
    """
    A Blender scene room generator consisting of floor, walls, and ceiling.
    """
    
    def __init__(self, area: float = 2000.0, room_placement_scale: float = 1.5, walls_height: float = 25.0,
                 background_images_folder: Path = None):
        """
        Initializes Room with given parameters.
        
        :param area: Total area of the room (X * Y).
        :type area: float
        :param room_placement_scale: Scale factor to make the room larger than the placement area.
        :type room_placement_scale: float
        :param walls_height: Height of the walls.
        :type walls_height: float
        :param background_images_folder: Directory containing background images to use instead of material packs.
        :type background_images_folder: Path
        """
        # Store room parameters
        self.area = area
        self.room_placement_scale = room_placement_scale
        self.walls_height = walls_height
        self.background_images_folder = background_images_folder
        
        
    def build(self):
        """
        Builds the room: creates floor, walls, and ceiling.
        """
        # Calculate placement area dimensions
        self.side_x_p, self.side_y_p = np.sqrt(self.area), np.sqrt(self.area)
        
        # Scale room by [room_placement_scale] to make it larger than placement area
        self.side_x_r = self.side_x_p * self.room_placement_scale
        self.side_y_r = self.side_y_p * self.room_placement_scale
        
        # Create Floor
        self.floor_obj = self.create_floor()
        # Compute room vertices in world coordinates
        self.room_vertices = [self.floor_obj.matrix_world @ v.co for v in self.floor_obj.data.vertices]
        
        # Create walls
        self.wall_objs = self.create_walls()
        
        # Create Ceiling
        self.ceiling_obj = self.create_ceiling()
        
        # Assign materials
        self.assign_materials()
        
        
    def create_floor(self):
        """
        Creates the floor of the room.
        
        :return floor_obj: The created floor object.
        :rtype: bpy.types.Object
        """
        bpy.ops.mesh.primitive_plane_add()
        floor_obj = bpy.context.active_object
        floor_obj.name = 'Floor'
        floor_obj.scale = (self.side_x_r * 0.5, self.side_y_r * 0.5, 1)
        bpy.ops.object.transform_apply(location = True, rotation = True, scale = True)
        return floor_obj
     
     
    def create_walls(self):
        """
        Creates the walls of the room as separate objects (one per wall face).
        
        :return wall_objs: List of created wall objects.
        :rtype: List[bpy.types.Object]
        """
        # Get floor mesh data and world matrix
        me = self.floor_obj.data
        mw = self.floor_obj.matrix_world
        
        # Create temporary bmesh from floor
        temp_bm = bmesh.new()
        temp_bm.from_mesh(me)
        temp_bm.edges.ensure_lookup_table()
        
        wall_objs = []
        wall_idx = 0

        # Iterate over boundary edges to create individual walls
        for edge in temp_bm.edges:
            
            if not edge.is_boundary:
                continue # Skip non-boundary edges
            
            # Get world coordinates of edge vertices
            v1_local = edge.verts[0].co
            v2_local = edge.verts[1].co
            v1 = mw @ v1_local
            v2 = mw @ v2_local
            # Compute top vertices by adding height
            v1_top = v1 + Vector((0.0, 0.0, self.walls_height))
            v2_top = v2 + Vector((0.0, 0.0, self.walls_height))

            # Create a new bmesh for this wall face
            wall_bm = bmesh.new()
            
            # Create vertices for this wall
            bv1 = wall_bm.verts.new(v1)
            bv2 = wall_bm.verts.new(v2)
            bv3 = wall_bm.verts.new(v2_top)
            bv4 = wall_bm.verts.new(v1_top)
            
            # Create face for this wall segment
            try:
                wall_bm.faces.new((bv1, bv2, bv3, bv4))
            except ValueError:
                wall_bm.free()
                continue
            
            # Finalize this wall's mesh
            wall_bm.normal_update()
            wall_mesh = bpy.data.meshes.new(f'Wall_{wall_idx}')
            wall_bm.to_mesh(wall_mesh)
            wall_bm.free()
            
            # Create object for this wall
            wall_obj = bpy.data.objects.new(f'Wall_{wall_idx}', wall_mesh)
            bpy.context.collection.objects.link(wall_obj)
            wall_objs.append(wall_obj)
            wall_idx += 1

        # Free temporary bmesh
        temp_bm.free()

        return wall_objs
    
    
    def create_ceiling(self):
        """
        Creates the ceiling of the room.
        
        :return ceiling_obj: The created ceiling object.
        :rtype: bpy.types.Object
        """
        bpy.ops.mesh.primitive_plane_add()
        ceiling_obj = bpy.context.active_object
        ceiling_obj.name = 'Ceiling'
        ceiling_obj.scale = (self.side_x_r * 0.5, self.side_y_r * 0.5, 1)
        ceiling_obj.location = (0.0, 0.0, self.walls_height)
        bpy.ops.object.transform_apply(location = True, rotation = True, scale = True)
        return ceiling_obj
    
    
    def show_placement_walls(self):
        """
        Makes room walls smaller (on the XY plane) to match placement area and taller (on the Z axis)
        to prevent objects from escaping the room.
        
        :return placement_vertices: List of vertices of the placement area in world coordinates.
        :rtype: List[Vector]
        """
        # Resize all wall objects to match placement area
        for wall_obj in self.wall_objs:
            wall_obj.scale = (1.0 / self.room_placement_scale, 1.0 / self.room_placement_scale, 10.0)
            
            # Select and make wall active for transform_apply
            bpy.ops.object.select_all(action = 'DESELECT')
            wall_obj.select_set(True)
            bpy.context.view_layer.objects.active = wall_obj
            bpy.ops.object.transform_apply(location = True, rotation = True, scale = True)
        
        # Compute placement area vertices in world coordinates
        self.placement_vertices = [v / self.room_placement_scale for v in self.room_vertices]
        
        # Return placement area vertices
        return self.placement_vertices
    
    
    def show_room_walls(self):
        """
        Makes room walls match room dimensions.
        
        :return room_vertices: List of vertices of the room in world coordinates.
        :rtype: List[Vector]
        """
        # Resize all wall objects to match room dimensions
        for wall_obj in self.wall_objs:
            wall_obj.scale = (1.0 * self.room_placement_scale, 1.0 * self.room_placement_scale, 1 / 10.0)
            
            # Select and make wall active for transform_apply
            bpy.ops.object.select_all(action = 'DESELECT')
            wall_obj.select_set(True)
            bpy.context.view_layer.objects.active = wall_obj
            bpy.ops.object.transform_apply(location = True, rotation = True, scale = True)
        
        # Return room vertices
        return self.room_vertices
    
    
    def _load_image(self, path: Path, colorspace: str):
        """
        Loads an image with error handling.

        :param path: Path to the image file.
        :type path: Path
        :param colorspace: Colorspace to set for the image.
        :type colorspace: str
        
        :return img: Loaded bpy.types.Image object.
        :rtype: bpy.types.Image
        """
        if not path.exists():
            raise FileNotFoundError(f'Texture file not found: {path}')
        try:
            img = bpy.data.images.load(str(path), check_existing = True)
            img.colorspace_settings.name = colorspace
            return img
        except Exception as e:
            raise RuntimeError(f'Failed to load image {path}: {e}')
        

    def _make_material_from_image(self, image_path: Path, obj_name: str):
        """
        Creates a simple Blender material from an image file with stretched (non-tiling) projection.
        Automatically selects appropriate coordinate system and rotation based on object name:
        - Walls (Wall_*): UV coordinates with 90° rotation for upright images
        - Floor: Generated coordinates without rotation
        
        :param image_path: Path to the image file.
        :type image_path: Path
        :param obj_name: Name of the object (used for material naming and determining coordinate system).
        :type obj_name: str
        
        :return mat: Created Blender material.
        :rtype: bpy.types.Material
        """
        # Create Blender material
        mat = bpy.data.materials.new(name = f'{obj_name}_{image_path.stem}')
        
        # Enable node-based material
        mat.use_nodes = True
        # Access the node tree
        nt = mat.node_tree 
        nodes = nt.nodes
        links = nt.links
        # Clear any default nodes
        nodes.clear()

        # Create Material Output node
        out = nodes.new('ShaderNodeOutputMaterial')
        # Create Principled BSDF node
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        # Connect them
        links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        
        # Create Texture Coordinate node
        tex_coord = nodes.new('ShaderNodeTexCoord')
        
        # Create texture node for color
        tex_color = nodes.new('ShaderNodeTexImage')
        tex_color.image = self._load_image(image_path, 'sRGB')
        tex_color.extension = 'EXTEND'  # Stretch the image to fill the entire surface without tiling
        
        # Apply coordinate system based on object type
        if obj_name.startswith('Wall_'):
            # Vertical walls need images rotated 90° clockwise to appear upright
            # Transformation (U, V) → (V, 1 - U) achieves this result
            
            # Create nodes to manipulate UV coordinates
            separate_uv = nodes.new('ShaderNodeSeparateXYZ')   # Splits (U, V) into separate U and V values
            combine_uv = nodes.new('ShaderNodeCombineXYZ')     # Recombines into new (U, V) vector
            invert_v = nodes.new('ShaderNodeMath')             # Computes 1 - input
            invert_v.operation = 'SUBTRACT'
            invert_v.inputs[0].default_value = 1.0             # First input = 1.0, second input will be U
            
            # Connect the transformation pipeline
            links.new(tex_coord.outputs['UV'], separate_uv.inputs['Vector'])
            
            # Perform coordinate swap and inversion
            links.new(separate_uv.outputs['Y'], combine_uv.inputs['X'])      # Old V becomes new U
            links.new(separate_uv.outputs['X'], invert_v.inputs[1])          # Old U goes to inverter
            links.new(invert_v.outputs['Value'], combine_uv.inputs['Y'])     # (1 - old U) becomes new V
            combine_uv.inputs['Z'].default_value = 0.0                       # Z is unused for 2D textures
            
            # Feed transformed coordinates to texture
            links.new(combine_uv.outputs['Vector'], tex_color.inputs['Vector'])
        else:
            # For floor, use Generated coordinates
            links.new(tex_coord.outputs['Generated'], tex_color.inputs['Vector'])
        
        # Connect base color texture to the shader's color input
        links.new(tex_color.outputs['Color'], bsdf.inputs['Base Color'])
        
        # Set roughness to a neutral value
        bsdf.inputs['Roughness'].default_value = 0.5

        return mat


    def assign_materials(self):
        """
        Assigns materials to floor and walls of a room using background images.
        """
        # Get all image files from background_images folder
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tga'}
        available_images = [f for f in self.background_images_folder.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not available_images:
            print(f'[WARN] No images found in {self.background_images_folder}')
            return
        
        # Calculate number of unique images needed (1 floor + N walls)
        n_images_needed = 1 + len(self.wall_objs)
        
        # Check if enough images are available
        if len(available_images) < n_images_needed:
            print(f'[WARN] Not enough unique images: need {n_images_needed}, found {len(available_images)}. Images may repeat.')
            # Sample with replacement if not enough images
            selected_images = [random.choice(available_images) for _ in range(n_images_needed)]
        else:
            # Sample without replacement to ensure uniqueness
            selected_images = random.sample(available_images, n_images_needed)
        
        # Assign material to floor (first selected image)
        try:
            image_path = selected_images[0]
            bpy.context.view_layer.objects.active = self.floor_obj
            blender_mat = self._make_material_from_image(image_path, self.floor_obj.name)
                
            if self.floor_obj.data.materials:
                self.floor_obj.data.materials[0] = blender_mat
            else:
                self.floor_obj.data.materials.append(blender_mat)
            
        except Exception as e:
            print(f'[UNEXPECTED] Error assigning image to {self.floor_obj.name}. {type(e).__name__}: {e}')
        
        # Assign different material to each wall (remaining selected images)
        for wall_idx, wall_obj in enumerate(self.wall_objs):
            try:
                # Use pre-selected image for this wall (offset by 1 for floor)
                image_path = selected_images[wall_idx + 1]
                
                # Make current wall active
                bpy.context.view_layer.objects.active = wall_obj
                
                # Simple UV unwrap - Smart UV Project works without viewport context
                bpy.ops.object.mode_set(mode = 'EDIT')
                bpy.ops.mesh.select_all(action = 'SELECT')
                bpy.ops.uv.smart_project(angle_limit = 66, island_margin = 0, scale_to_bounds = True)
                bpy.ops.object.mode_set(mode = 'OBJECT')
                
                # Create and apply material from image (stretched, no tiling)
                blender_mat = self._make_material_from_image(image_path, wall_obj.name)
                    
                if wall_obj.data.materials:
                    wall_obj.data.materials[0] = blender_mat
                else:
                    wall_obj.data.materials.append(blender_mat)
                
            except Exception as e:
                print(f'[UNEXPECTED] Error assigning image to {wall_obj.name}. {type(e).__name__}: {e}')