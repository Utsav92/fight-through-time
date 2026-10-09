import bpy
import os
import math

BASE_DIR = r"C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time"
FRAMES_DIR = os.path.join(BASE_DIR, "data", "frames_combined")
BLEND_OUT = os.path.join(BASE_DIR, "blender", "fight_through_time.blend")
os.makedirs(os.path.dirname(BLEND_OUT), exist_ok=True)

print("Initializing Blender scene: FIGHT THROUGH TIME — INTERACTIVE 4D MOTION SCULPTURE...")
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.name = "4D_Motion_Sculpture"

# Render Settings
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.fps = 25
scene.frame_start = 1
scene.frame_end = 450

# World Background
world = bpy.data.worlds.new("Void_World")
world.use_nodes = True
bg_node = world.node_tree.nodes.get("Background")
if bg_node:
    bg_node.inputs["Color"].default_value = (0.015, 0.018, 0.025, 1.0)
    bg_node.inputs["Strength"].default_value = 0.5
scene.world = world

# Lighting
key_light_data = bpy.data.lights.new("Key_Light", type="SUN")
key_light_data.energy = 2.5
key_light_data.color = (0.9, 0.95, 1.0)
key_light = bpy.data.objects.new("Key_Light", key_light_data)
key_light.location = (5.0, -8.0, 10.0)
key_light.rotation_euler = (math.radians(45), 0, math.radians(30))
scene.collection.objects.link(key_light)

cyan_rim_data = bpy.data.lights.new("Cyan_Rim", type="POINT")
cyan_rim_data.energy = 500.0
cyan_rim_data.color = (0.0, 0.9, 1.0) # Bruce Lee cool aura
cyan_rim = bpy.data.objects.new("Cyan_Rim", cyan_rim_data)
cyan_rim.location = (-4.0, 3.0, 2.0)
scene.collection.objects.link(cyan_rim)

amber_rim_data = bpy.data.lights.new("Amber_Rim", type="POINT")
amber_rim_data.energy = 500.0
amber_rim_data.color = (1.0, 0.55, 0.0) # Chuck Norris warm aura
amber_rim = bpy.data.objects.new("Amber_Rim", amber_rim_data)
amber_rim.location = (4.0, 3.0, 2.0)
scene.collection.objects.link(amber_rim)

# Reflective Ground Plane
bpy.ops.mesh.primitive_plane_add(size=40, location=(0, 0, -4.0))
floor = bpy.context.active_object
floor.name = "Floor_Reflective_Grid"
floor_mat = bpy.data.materials.new("Floor_Material")
floor_mat.use_nodes = True
f_bsdf = floor_mat.node_tree.nodes.get("Principled BSDF")
if f_bsdf:
    f_bsdf.inputs["Base Color"].default_value = (0.02, 0.025, 0.035, 1.0)
    f_bsdf.inputs["Roughness"].default_value = 0.25
    f_bsdf.inputs["Metallic"].default_value = 0.4
floor.data.materials.append(floor_mat)

# Camera Reveal Animation
cam_data = bpy.data.cameras.new("Cinematic_4D_Camera")
cam_data.lens = 38 # mm
cam_obj = bpy.data.objects.new("Cinematic_4D_Camera", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

# Camera Target Empty
target_empty = bpy.data.objects.new("Camera_Target", None)
target_empty.location = (0.0, 1.0, -4.0)
scene.collection.objects.link(target_empty)

track_constraint = cam_obj.constraints.new(type="TRACK_TO")
track_constraint.target = target_empty
track_constraint.track_axis = "TRACK_NEGATIVE_Z"
track_constraint.up_axis = "UP_Y"

# Animate Camera Path across 450 frames
keyframes = [
    (1, 0.0, 8.8, 1.0),      # 0° Chronophotography front view
    (60, 0.0, 8.8, 1.0),     # Baseline hold
    (180, 45.0, 8.5, 1.7),   # 45° 4D Spatial Reveal
    (250, 90.0, 8.5, 2.0),   # 90° Time Axis Profile
    (360, 20.0, 8.0, 1.4),   # 4D Freeze Frame Orbit
    (450, 35.0, 8.2, 1.5)    # Final dynamic angle
]

for frame_num, deg, dist, height in keyframes:
    rad = math.radians(deg)
    cx = math.sin(rad) * dist
    cz = math.cos(rad) * dist - 4.0
    cy = height
    cam_obj.location = (cx, cy, cz)
    cam_obj.keyframe_insert(data_path="location", frame=frame_num)

# 7 Temporal States Setup
OPACITIES = [1.0, 0.82, 0.68, 0.55, 0.43, 0.32, 0.22]
TIME_DEPTH = 1.35 # meters per 0.4s
tex_sample_path = os.path.join(FRAMES_DIR, "frame_0420.png")

sculpture_collection = bpy.data.collections.new("4D_Temporal_Sculpture")
scene.collection.children.link(sculpture_collection)

for k in range(7):
    z_k = -k * TIME_DEPTH
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0, 1.0, z_k))
    plane = bpy.context.active_object
    plane.name = f"Temporal_State_T_minus_{k*0.4:.1f}s"
    plane.scale = (4.37, 2.0, 1.0)
    plane.rotation_euler = (math.radians(90), 0, 0)
    
    # Material
    mat = bpy.data.materials.new(f"Mat_Temporal_State_{k}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    
    if os.path.exists(tex_sample_path):
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.image = bpy.data.images.load(tex_sample_path)
        links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
        
        # Transparent opacity ladder
        opac = OPACITIES[k]
        mult_node = nodes.new("ShaderNodeMath")
        mult_node.operation = "MULTIPLY"
        mult_node.inputs[1].default_value = opac
        links.new(tex_node.outputs["Alpha"], mult_node.inputs[0])
        links.new(mult_node.outputs["Value"], bsdf.inputs["Alpha"])
        
        # Subtle emission for temporal energy
        if k > 0:
            emit_col = (0.0, 0.8, 1.0, 1.0) if k % 2 == 1 else (1.0, 0.5, 0.0, 1.0)
            bsdf.inputs["Emission Color"].default_value = emit_col
            bsdf.inputs["Emission Strength"].default_value = 0.2 * (1.0 - k/7.0)
            
    mat.blend_method = "BLEND"
    plane.data.materials.append(mat)
    
    # Relink into sculpture collection
    scene.collection.objects.unlink(plane)
    sculpture_collection.objects.link(plane)

# Save .blend project
bpy.ops.wm.save_as_mainfile(filepath=BLEND_OUT)
print(f"SUCCESS: Saved Blender project to {BLEND_OUT}")
