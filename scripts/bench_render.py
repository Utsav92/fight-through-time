import bpy, os, time

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items] else 'BLENDER_EEVEE'
scene.render.resolution_x = 1280
scene.render.resolution_y = 720
scene.render.resolution_percentage = 100

base_dir = r'C:\Users\Utsav\.gemini\antigravity\scratch\fight-through-time'
out_path = os.path.join(base_dir, 'samples', 'bench_render.png')
scene.render.filepath = out_path

# Camera
cam_data = bpy.data.cameras.new('Camera')
cam_obj = bpy.data.objects.new('Camera', cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_obj.location = (0, -6, 2)
cam_obj.rotation_euler = (1.2, 0, 0)

# Add textured plane
bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,1))
plane = bpy.context.active_object
mat = bpy.data.materials.new('TestMat')
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
bsdf = nodes.get('Principled BSDF')

tex_path = os.path.join(base_dir, 'data', 'frames_combined', 'frame_0420.png')
if os.path.exists(tex_path):
    tex_node = nodes.new('ShaderNodeTexImage')
    tex_node.image = bpy.data.images.load(tex_path)
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(tex_node.outputs['Alpha'], bsdf.inputs['Alpha'])
    mat.blend_method = 'BLEND'
plane.data.materials.append(mat)

t0 = time.time()
bpy.ops.render.render(write_still=True)
t1 = time.time()
print(f'EEVEE frame render took: {t1-t0:.3f}s')
