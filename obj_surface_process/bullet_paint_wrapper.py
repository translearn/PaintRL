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


def _show_flange_debug_line(points):
    # only for debug purpose, robot pose may change afterwards
    robot_pose = getLinkState(2, 6)[0]
    for point in points:
        addUserDebugLine(robot_pose, point, [0, 1, 0])


def _show_texture_image(pixels, width, height):
    pixel_array = np.reshape(np.asarray(pixels), (width, height, 3))
    img = Image.fromarray(pixel_array, 'RGB')
    img.show()


class BarycentricInterpolator:
    """
    Each f line in the obj will be initialized as a barycentric interpolator
    Algorithm taken from:
    https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    """

    def __init__(self, a, b, c):
        self._a = a
        self._b = b
        self._c = c
        self._v0 = np.subtract(b, a)
        self._v1 = np.subtract(c, a)
        self._d00 = np.dot(self._v0, self._v0)
        self._d01 = np.dot(self._v0, self._v1)
        self._d11 = np.dot(self._v1, self._v1)
        self._inv_denom = 1.0 / (self._d00 * self._d11 - self._d01 * self._d01)

        self._uva = None
        self._uvb = None
        self._uvc = None

    def _get_bary_coordinate(self, point):
        v2 = np.subtract(point, self._a)
        d20 = np.dot(v2, self._v0)
        d21 = np.dot(v2, self._v1)
        v = (self._d11 * d20 - self._d01 * d21) * self._inv_denom
        w = (self._d00 * d21 - self._d01 * d20) * self._inv_denom
        u = 1.0 - v - w
        return u, v, w

    def is_inside_triangle(self, point):
        bary_a, bary_b, bary_c = self._get_bary_coordinate(point)
        return -0.1 < bary_a < 1.1 and -0.1 < bary_b < 1.1 and -0.1 < bary_c < 1.1

    def set_uv_coordinate(self, uva, uvb, uvc):
        self._uva = uva
        self._uvb = uvb
        self._uvc = uvc

    def get_texel(self, point):
        bary_a, bary_b, bary_c = self._get_bary_coordinate(point)
        # For efficiency reason, do not check uvs are set or not
        return list(np.dot(bary_a, self._uva) + np.dot(bary_b, self._uvb) + np.dot(bary_c, self._uvc))

    def add_debug_info(self):
        addUserDebugLine(self._a, self._b, [1, 0, 0])
        addUserDebugLine(self._b, self._c, [1, 0, 0])
        addUserDebugLine(self._c, self._a, [1, 0, 0])


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

    def paint(self, points, color):
        color = [np.uint8(i * 255) for i in color]
        nearest_vertices = self.vertices_kd_tree.query(points, k=5)[1]
        debug = []
        not_found = []
        for i, point in enumerate(points):
            baries = [k for j in range(len(nearest_vertices[i])) for k in self.uv_map[nearest_vertices[i][j]]]
            found = False
            for bary in baries:
                if bary.is_inside_triangle(point):
                    u, v = bary.get_texel(point)
                    # pixels_coord = uv_coord * [width, height], normal round up, without Bilinear filtering
                    # some uv coordinate are not in range [0, 1], therefore mode the calculated value
                    i = int(round(self.texture_width * u)) % self.texture_width
                    j = int(round(self.texture_height * v)) % self.texture_height
                    texel = (i + j * self.texture_width) * 3
                    debug.append(texel)
                    self.texture_pixels[texel] = color[0]
                    self.texture_pixels[texel + 1] = color[1]
                    self.texture_pixels[texel + 2] = color[2]
                    found = True
                    # break
            if not found:
                _show_flange_debug_line([point])
                for bary in baries:
                    bary.is_inside_triangle(point)
                    bary.add_debug_info()
                not_found.append(point)

        _show_flange_debug_line(not_found)

        # _show_flange_debug_line([self.vertices_kd_tree.data[i] for i in extracted_indexes])
        _show_texture_image(self.texture_pixels, self.texture_width, self.texture_height)

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


def _get_global_coordinate(urdf_id, points):
    base_pose, base_orn = getBasePositionAndOrientation(urdf_id)
    local_points = []
    for point in points:
        pose, _ = multiplyTransforms(base_pose, base_orn, point, [0, 0, 0, 1])
        local_points.append(pose)
    return local_points


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
                coordinate = []
                for v in content[1:]:
                    coordinate.append(float(v))
                v_array.append(coordinate)
            elif content[0] == 'vt':
                vt_array.append([float(content[1]), 1 - float(content[2])])

        global_v_array = _get_global_coordinate(urdf_obj.urdf_id, v_array)
        f.seek(0)
        uv_map = {}
        for line in f:
            content = line.split()
            if not len(content):
                continue
            if content[0] == 'f':
                triangle_point_indexes = [int(i.split('/')[0]) - 1 for i in content[1:]]
                v_coordinates = [global_v_array[i] for i in triangle_point_indexes]
                uv_coordinates = [vt_array[int(i.split('/')[1]) - 1] for i in content[1:]]
                bary_interpolator = BarycentricInterpolator(*v_coordinates)
                bary_interpolator.set_uv_coordinate(*uv_coordinates)
                for v_index in triangle_point_indexes:
                    if v_index not in uv_map:
                        uv_map[v_index] = []
                    uv_map[v_index].append(bary_interpolator)

        urdf_obj.vertices_kd_tree = cKDTree(global_v_array)
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
