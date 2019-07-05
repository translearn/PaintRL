#! /usr/bin/python
"""
Please make sure that Blender is accessible in terminal
"""
import os
import tempfile
from subprocess import Popen, PIPE
import argparse


def _add_quote(string):
    return "'" + string + "'"


def process_obj_file(obj_path, texture_path, output_path):

    blender_script = """
import bpy
from mathutils import *
from math import *

material_name = 'new_material'

# delete the default created objects
if len(bpy.context.selected_objects):
    bpy.ops.object.delete()

# load file
bpy.ops.import_scene.obj(filepath={0})
obj = bpy.context.selected_objects[0]

# resize the object
size = obj.dimensions
std_scale = Vector((1.0, 1.0, 1.0))
if not 0.5 <= max(size) <= 1.5:
    scale = max(size) / 1
    bpy.ops.transform.resize(value=std_scale/scale, proportional='ENABLED')
    
# create new material and link it to the object
material = bpy.data.materials.new(material_name)
obj.data.materials.append(material)

uv_texture = bpy.data.textures.new('new_texture', type='IMAGE')
uv_texture.image = bpy.data.images.load({1})

bpy.data.materials[material_name].texture_slots.add()
bpy.data.materials[material_name].active_texture = uv_texture

# create smart uv project
bpy.context.scene.objects.active = obj
bpy.ops.uv.smart_project()

# export obj file with uv mapping
bpy.ops.export_scene.obj(filepath={2}, use_selection=True)
""".format(_add_quote(obj_path), _add_quote(texture_path), _add_quote(output_path))

    with tempfile.NamedTemporaryFile(mode='w') as tmp:
        tmp.write(blender_script)
        blender_cmd = Popen(['blender', '--background', '--python', tmp.name], stdout=PIPE)
        tmp.flush()
        result = blender_cmd.communicate()
        print(result[0].decode('ascii'))


if __name__ == '__main__':
    default_root = os.path.dirname(os.path.realpath(__file__))
    default_root = os.path.join(os.path.dirname(default_root), 'urdf/painting/')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-path', default=default_root + 'carDoor.obj')
    parser.add_argument('-t', '--texture-path', default=default_root + 'pattern.jpg')
    parser.add_argument('-o', '--output-path', default=default_root + 'test.obj')
    args = parser.parse_args()
    process_obj_file(args.input_path, args.texture_path, args.output_path)
