"""
This file wraps the urdf loading of pybullet, adding additional logic to process the uv mapping,
please make sure the mesh obj files are processed by the process_script.py
the changeTexture function will be called implicitly as long as the new method paint is called
"""
import os
import pybullet as p
from pybullet import *
import pybullet_data
import xml.etree.ElementTree as Et
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

BULLET_LIB_PATH = pybullet_data.getDataPath()
_urdf_cache = {}


class URDF:
    """
    Store the loaded urdf cache and its correspondent change texture parameters
    """
    def __init__(self, urdf_id=-1):
        self.urdf_id = urdf_id
        self.uv_map = None
        self.vertices_kd_tree = None
        self.texture_id = None
        self.texture_width = None
        self.texture_height = None
        self.texture_pixels = None
        self.inverted_pose, self.inverted_orn = invertTransform(*getBasePositionAndOrientation(self.urdf_id))

    def _get_local_coordinate(self, points):
        local_points = []
        for point in points:
            pose, _ = multiplyTransforms(self.inverted_pose, self.inverted_orn, point, [0, 0, 0, 1])
            local_points.append(pose)
        return local_points

    def paint(self, points, color):
        color = [i * 255 for i in color]
        points = self._get_local_coordinate(points)
        nearest_neighbors = self.vertices_kd_tree.query(points, k=3)
        related_indexes = nearest_neighbors[1]

        # # debug
        # base_pose, base_orn = getBasePositionAndOrientation(self.urdf_id)
        # robot_pose = getLinkState(2, 6)[0]
        # for index_set in nearest_neighbors[1]:
        #     for index in index_set:
        #         # for point in point_set:
        #         #     # curr_pose, curr_orn = multiplyTransforms()
        #         point, _ = multiplyTransforms(base_pose, base_orn, self.vertices_kd_tree.data[index], [0, 0, 0, 1])
        #         p.addUserDebugLine(robot_pose, point, [0, 1, 0])

        extracted_indexes = list(set([j for i in related_indexes for j in i]))
        before = np.array(self.texture_pixels, copy=True)
        for index in extracted_indexes:
            pixel = self.uv_map[index][0]
            self.texture_pixels[pixel] = color[0]
            self.texture_pixels[pixel + 1] = color[1]
            self.texture_pixels[pixel + 2] = color[2]

        # debug
        # pixel_array = np.reshape(np.asarray(self.texture_pixels), (240, 240, 3))
        # pixel_array = np.asarray(self.texture_pixels).reshape((240, 240, 3))
        pixel_array = before.reshape((240, 240, 3))
        img = Image.fromarray(pixel_array, 'RGB')
        img.show()

        changeTexture(self.texture_id, self.texture_pixels, self. texture_width, self.texture_height)


def _get_abs_file_path(root_path, path):
    # the file path in mtl could be a relative path or an absolute path
    full_path = os.path.join(root_path, path)
    if os.path.isfile(path):
        return path
    elif os.path.isfile(full_path):
        return full_path
    else:
        return None


def _retrieve_related_file_path(file_path):
    root_path = os.path.dirname(file_path)
    urdf = Et.parse(file_path).getroot()
    targets = urdf.findall('./link/visual/geometry/mesh')
    if targets:
        obj_file_path = targets[0].get('filename')
        # The textured obj file normally has a mtl file with the same name, check this to avoid operating raw obj
        obj_file_name, extension_name = os.path.splitext(obj_file_path)
        mtl_file_path = _get_abs_file_path(root_path, obj_file_name + '.mtl')

        if extension_name != '.obj' or not mtl_file_path:
            return None, None
        with open(mtl_file_path, mode='r') as f:
            for line in f:
                if 'map_Kd' in line:
                    abs_obj_path = _get_abs_file_path(root_path, obj_file_path)
                    abs_texture_path = _get_abs_file_path(root_path, line.split(' ')[-1].strip())
                    return abs_obj_path, abs_texture_path
            return None, None
    else:
        return None, None


def _search_urdf_path(file_path):
    path = file_path
    if not path:
        raise TypeError('file path is not given!')
    if not os.path.isfile(path):
        path = os.path.join(BULLET_LIB_PATH, path)
        setAdditionalSearchPath(BULLET_LIB_PATH)
        if not os.path.isfile(path):
            raise TypeError('file does not exist!')
    return path


def _cache_texture(urdf_obj, obj_path, texture_path):
    with Image.open(texture_path) as img:
        width, height = img.size
        pixels = list(np.asarray(img.convert('RGB')).ravel())
        texture_id = loadTexture(texture_path)
        urdf_obj.texture_id = texture_id
        urdf_obj.texture_pixels = pixels
        urdf_obj.texture_width = width
        urdf_obj.texture_height = height
        # after this operation, the texture could be changed by override the color values in pixels
        changeVisualShape(urdf_obj.urdf_id, -1, textureUniqueId=texture_id)
        _cache_obj(urdf_obj, obj_path)


def _cache_obj(urdf_obj, obj_path):
    # assume strictly that v comes first, then vt, then f.
    v_array = []
    vt_array = []
    with open(obj_path, mode='r') as f:
        for line in f:
            content = line.split()
            if not len(content):
                continue
            if content[0] == 'v':
                v_array.append(content[1:])
            elif content[0] == 'vt':
                # pixels_coord = uv_coord * [width, height], normal round up, without Bilinear filtering
                # some uv coordinate are not in range [0, 1], therefore mode the calculated value
                i = int(round(urdf_obj.texture_width * float(content[1]))) % urdf_obj.texture_width
                j = int(round(urdf_obj.texture_height * float(content[2]))) % urdf_obj.texture_height
                # linear position of the pixel values, 0 -> R, +1 -> G, +2 -> B
                vt_array.append((i + j * urdf_obj.texture_width) * 3)
        f.seek(0)
        uv_map = np.zeros(shape=[len(v_array), 1], dtype=int)
        for line in f:
            content = line.split()
            if not len(content):
                continue
            if content[0] == 'f':
                for item in content[1:]:
                    temp = item.split('/')
                    uv_map[int(temp[0]) - 1] = vt_array[int(temp[1]) - 1]

        urdf_obj.vertices_kd_tree = cKDTree(v_array)
        urdf_obj.uv_map = uv_map


def _load_urdf_wrapper(*args, **kwargs):
    try:
        path = _search_urdf_path(args[0])
        u_id = p.loadURDF(*args, **kwargs)
        obj_file_path, texture_file_path = _retrieve_related_file_path(path)
        # Texture file exists, prepare for texture manipulation
        if texture_file_path:
            urdf = URDF(u_id)
            _urdf_cache[u_id] = urdf
            _cache_texture(urdf, obj_file_path, texture_file_path)
        return u_id
    except error as e:
        print(str(e))
        return -1


loadURDF = _load_urdf_wrapper


def paint(urdf_id, points, color):
    """
    paint a specific part
    :param urdf_id: integer ID of the model returned by bullet
    :param points: intersection points in global coordinate
    :param color: list [r, g, b], each in range [0, 1]
    :return:
    """
    _urdf_cache[urdf_id].paint(points, color)
