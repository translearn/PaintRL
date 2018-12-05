"""
This file wraps the urdf loading of pybullet as a new function for painting parts, adding additional
logic to process the uv mapping, please make sure the mesh obj files are processed by the process_script.py
the changeTexture function will be called implicitly as long as the new method paint is called
"""
import os
import enum
from pybullet import *
import xml.etree.ElementTree as Et
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

_urdf_cache = {}
AXES = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]


def _show_flange_debug_line(points):
    # only for debug purpose, robot pose may change afterwards
    robot_pose = getLinkState(2, 6)[0]
    for point in points:
        addUserDebugLine(robot_pose, point, (0, 1, 0))


def _get_texture_image(pixels, width, height):
    pixel_array = np.reshape(np.asarray(pixels), (width, height, 3))
    img = Image.fromarray(pixel_array, 'RGB')
    return img


def _get_color(color):
    return [np.uint8(i * 255) for i in color]


class Side(enum.Enum):
    """First define only two sides for each part"""
    front = 1
    back = 2
    other = 3


VALID_SIDE = [Side.front, Side.back]


def _get_pixel_coordinate(u, v, width, height):
    # pixels_coord = uv_coord * [width, height], normal round up, without Bilinear filtering
    # some uv coordinate are not in range [0, 1]
    i = min(int(round(width * u)), width - 1)
    j = min(int(round(height * v)), height - 1)
    return i, j


def _get_point_along_normal(point, length, vn):
    proportional_vn = [i * length for i in vn]
    return [a + b for a, b in zip(point, proportional_vn)]


class BarycentricInterpolator:
    """
    Each f line in the obj will be initialized as a barycentric interpolator
    The interpolation algorithm is taken from:
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

        self._vn = None
        self._side = None

    def _get_bary_coordinate(self, point):
        v2 = np.subtract(point, self._a)
        d20 = np.dot(v2, self._v0)
        d21 = np.dot(v2, self._v1)
        v = (self._d11 * d20 - self._d01 * d21) * self._inv_denom
        w = (self._d00 * d21 - self._d01 * d20) * self._inv_denom
        u = 1.0 - v - w
        return u, v, w

    def get_min_uvw(self, point):
        bary_a, bary_b, bary_c = self._get_bary_coordinate(point)
        return min(bary_a, bary_b, bary_c)

    def is_inside_triangle(self, point):
        bary_a, bary_b, bary_c = self._get_bary_coordinate(point)
        return 0 <= bary_a <= 1 and 0 <= bary_b <= 1 and 0 <= bary_c <= 1

    def set_uv_coordinate(self, uva, uvb, uvc):
        self._uva = uva
        self._uvb = uvb
        self._uvc = uvc

    def get_uv_pixels(self, width, height):
        pixels = []
        uva = _get_pixel_coordinate(*self._uva, width, height)
        uvb = _get_pixel_coordinate(*self._uvb, width, height)
        uvc = _get_pixel_coordinate(*self._uvc, width, height)
        pixels.append(uva)
        pixels.append(uvb)
        pixels.append(uvc)
        # Here make a 2D triangle, use barycentric coordinate to judge if pixels inside the triangle
        uv_bary = BarycentricInterpolator(self._uva, self._uvb, self._uvc)
        x_min, x_max = min(uva[0], uvb[0], uvc[0]), max(uva[0], uvb[0], uvc[0])
        y_min, y_max = min(uva[1], uvb[1], uvc[1]), max(uva[1], uvb[1], uvc[1])
        for u in range(x_min, x_max + 1):
            for v in range(y_min, y_max + 1):
                if uv_bary.is_inside_triangle((u / width, v / height)):
                    pixels.append((u, v))
        return pixels

    def set_face_normal(self, vn, front_normal):
        self._vn = vn
        # normal could be recalculated or just use the value from obj file, here recalculated
        self.align_normal()
        self._side = _get_side(self._vn, front_normal)

    def is_in_same_side(self, side):
        return self._side == side

    def get_side(self):
        return self._side

    def get_texel(self, point, width, height):
        bary_a, bary_b, bary_c = self._get_bary_coordinate(point)
        # For efficiency reason, do not check uvs are set or not
        u, v = list(np.dot(bary_a, self._uva) + np.dot(bary_b, self._uvb) + np.dot(bary_c, self._uvc))
        return _get_pixel_coordinate(u, v, width, height)

    def add_debug_info(self):
        # draw the triangle for debug
        color = (1, 0, 0)
        addUserDebugLine(self._a, self._b, color)
        addUserDebugLine(self._b, self._c, color)
        addUserDebugLine(self._c, self._a, color)

    def get_point_along_normal(self, point, length):
        return _get_point_along_normal(point, length, self._vn)

    def get_normal(self):
        return self._vn

    def draw_face_normal(self):
        center_point = []
        for a, b, c in zip(self._a, self._b, self._c):
            center_point.append((a + b + c) / 3)
        target_point = self.get_point_along_normal(center_point, 0.25)
        addUserDebugLine(center_point, target_point, (1, 0, 0) if self._side == Side.front else (0, 1, 0))

    def calculate_normal_from_abc(self):
        # Trust the sequence in obj file, that the points are given in counter clockwise sequence
        u = [b - a for a, b in zip(self._a, self._b)]
        v = [c - a for a, c in zip(self._a, self._c)]
        n = np.cross(u, v)
        norm_n = np.linalg.norm(n)
        n = [i / norm_n for i in n]
        return n

    def align_normal(self):
        self._vn = self.calculate_normal_from_abc()


class Part:
    HOOK_DISTANCE_TO_PART = 0.1
    # Color to mark irrelevant pixels, used for preprocessing and calculate rewards
    IRRELEVANT_COLOR = (1, 1, 1)
    FRONT_COLOR = (0.75, 0.75, 0.75)
    BACK_COLOR = (0, 1, 0)
    # Place holder in KD-tree, no point can have such a coordinate
    IRRELEVANT_POSE = (10, 10, 10)
    """
    Store the loaded urdf cache and its correspondent change texture parameters,
    extract the pixels on the part to be painted.
    """
    def __init__(self, urdf_id=-1):
        self.urdf_id = urdf_id
        self.uv_map = None
        self.vertices = None
        self.vertices_kd_tree = {}
        self.texture_id = None
        self.texture_width = None
        self.texture_height = None
        self.texture_pixels = None
        self._start_points = {}
        self.profile = {}
        self.principle_axes = None
        self.front_normal = None

    def _get_texel(self, i, j):
        return (i + j * self.texture_width) * 3

    def _change_pixel(self, color, i, j):
        texel = self._get_texel(i, j)
        self.texture_pixels[texel] = color[0]
        self.texture_pixels[texel + 1] = color[1]
        self.texture_pixels[texel + 2] = color[2]

    def _change_texel_color(self, color, bary, point):
        i, j = bary.get_texel(point, self.texture_width, self.texture_height)
        self._change_pixel(color, i, j)

    def _get_closest_bary(self, point, nearest_vertex, side):
        closest_uvw = -1
        closest_bary = None
        for bary in self.uv_map[nearest_vertex]:
            if not bary.is_in_same_side(side):
                continue
            if bary.is_inside_triangle(point):
                return bary
            else:
                if not closest_bary:
                    closest_bary = bary
                min_uvw = bary.get_min_uvw(point)
                if min_uvw >= closest_uvw:
                    closest_uvw = min_uvw
                    closest_bary = bary
        return closest_bary

    def _get_hook_point(self, point, side):
        nearest_vertex = self.vertices_kd_tree[side].query(point, k=1)[1]
        bary = self._get_closest_bary(point, nearest_vertex, side)
        if bary:
            pose = bary.get_point_along_normal(point, Part.HOOK_DISTANCE_TO_PART)
            orn = [-i for i in bary.get_normal()]
            # bary.add_debug_info()
            # bary.draw_face_normal()
            return pose, orn
        return None, None

    def paint(self, points, color, side):
        color = _get_color(color)
        # current_side = _get_side([-i for i in side], self.front_normal)
        current_side = side
        nearest_vertices = self.vertices_kd_tree[current_side].query(points, k=1)[1]
        for i, point in enumerate(points):
            bary = self._get_closest_bary(point, nearest_vertices[i], current_side)
            if bary:
                self._change_texel_color(color, bary, point)
                # for bary in self.uv_map[nearest_vertices[i]]:
                #     bary.add_debug_info()
                #     bary.draw_face_normal()
        # _get_texture_image(self.texture_pixels, self.texture_width, self.texture_height).show()
        changeTexture(self.texture_id, self.texture_pixels, self. texture_width, self.texture_height)

    def _label_part(self):
        # preprocessing all irrelevant pixels
        target_pixels = [(i, j) for i in range(self.texture_width) for j in range(self.texture_height)]
        for side in self.profile:
            target_pixels = [i for i in target_pixels if i not in self.profile[side]]
        irr_color = _get_color(Part.IRRELEVANT_COLOR)
        front_color = _get_color(Part.FRONT_COLOR)
        back_color = _get_color(Part.BACK_COLOR)
        for point in target_pixels:
            self._change_pixel(irr_color, *point)
        for point in self.profile[Side.back]:
            self._change_pixel(back_color, *point)
        for point in self.profile[Side.front]:
            self._change_pixel(front_color, *point)
        # _get_texture_image(self.texture_pixels, self.texture_width, self.texture_height).show()

    def _build_kd_tree(self):
        side_label = {}
        point_list = {}
        for side in self.profile:
            point_list[side] = [i for i in self.vertices]

        for key, p_map in self.uv_map.items():
            for side in self.profile:
                side_label[side] = False
            for bary in p_map:
                side_label[bary.get_side()] = True
            for side, label in side_label.items():
                if not label:
                    point_list[side][key] = Part.IRRELEVANT_POSE
        for side in self.profile:
            self.vertices_kd_tree[side] = cKDTree(point_list[side])

    def preprocess(self):
        """
        Store relevant pixels according to its side of the part into self.profile
        Mark irrelevant pixels to IRRELEVANT_COLOR
        """
        for _, p_map in self.uv_map.items():
            for bary in p_map:
                if bary.get_side():
                    pixels = bary.get_uv_pixels(self.texture_width, self.texture_height)
                    # bary.add_debug_info()
                    side = bary.get_side()
                    if side not in self.profile:
                        self.profile[side] = []
                    self.profile[side].extend(pixels)
        if self.profile:
            invalid_side = None
            for side in self.profile:
                if side not in VALID_SIDE:
                    invalid_side = side
                    continue
                self.profile[side] = list(set(self.profile[side]))
            del self.profile[invalid_side]
            self._label_part()
            self._build_kd_tree()

    def get_texture_size(self):
        return self.texture_width, self.texture_height

    def get_job_status(self, side, color):
        color = _get_color(color)
        finished_counter = 0
        for pixel in self.profile[side]:
            texel = self._get_texel(*pixel)
            texel_color = self.texture_pixels[texel:texel + 3]
            if color == texel_color:
                finished_counter += 1
        return finished_counter

    def get_job_limit(self, side):
        return len(self.profile[side])

    def get_texture_image(self):
        return _get_texture_image(self.texture_pixels, self.texture_width, self.texture_height)

    def set_start_points(self, corner_points):
        for side in VALID_SIDE:
            self._start_points[side] = []
        for i, point in enumerate(corner_points):
            for side in VALID_SIDE:
                pose, orn = self._get_hook_point(point, side)
                if pose:
                    self._start_points[side].append([pose, orn])

    def get_start_points(self, side):
        return self._start_points[side]

    def get_guided_point(self, point, normal, delta_axis1, delta_axis2):
        point = list(point)
        point[self.principle_axes[0]] += delta_axis1
        point[self.principle_axes[1]] += delta_axis2
        current_side = _get_side([-i for i in normal], self.front_normal)
        end_point = [a + b for a, b in zip(point, normal)]
        result = rayTestBatch([point], [end_point])
        if not result[0][0] == self.urdf_id:
            print('Error in Ray Test!!!')
            return None, normal
        surface_point = result[0][3]
        pos, orn = self._get_hook_point(surface_point, current_side)
        return pos, orn if orn else normal


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
        pose, _ = multiplyTransforms(base_pose, base_orn, point, (0, 0, 0, 1))
        local_points.append(pose)
    return local_points


def _get_coordinate_from_line(content, v_array):
    coordinate = []
    for v in content[1:]:
        coordinate.append(float(v))
    v_array.append(coordinate)


def _retrieve_obj_elements(file):
    file.seek(0)
    v_array = []
    vt_array = []
    vn_array = []
    for line in file:
        content = line.split()
        if not len(content):
            continue
        if content[0] == 'v':
            _get_coordinate_from_line(content, v_array)
        elif content[0] == 'vt':
            vt_array.append([float(content[1]), 1 - float(content[2])])
        elif content[0] == 'vn':
            _get_coordinate_from_line(content, vn_array)
    return v_array, vn_array, vt_array


def _get_included_angle(a, b):
    # assume unified vector
    return np.arccos(np.dot(a, b))


def _get_side(front_normal, v):
    # Take care of the possible rotation made in loading the part!
    angle_front = _get_included_angle(front_normal, v)
    back_normal = [-i for i in front_normal]
    angle_back = _get_included_angle(back_normal, v)
    if - np.pi / 3 <= angle_front <= np.pi / 3:
        return Side.front
    if - np.pi / 3 <= angle_back <= np.pi / 3:
        return Side.back
    return Side.other


def _get_uv_map(file, v_array, vt_array, vn_array, front_normal):
    file.seek(0)
    uv_map = {}
    for line in file:
        content = line.split()
        if not len(content):
            continue
        if content[0] == 'f':
            triangle_point_indexes = [int(i.split('/')[0]) - 1 for i in content[1:]]
            v_coordinates = [v_array[i] for i in triangle_point_indexes]
            uv_coordinates = [vt_array[int(i.split('/')[1]) - 1] for i in content[1:]]
            bary_interpolator = BarycentricInterpolator(*v_coordinates)
            bary_interpolator.set_uv_coordinate(*uv_coordinates)
            vn_index = [int(i.split('/')[2]) - 1 for i in content[1:] if len(i.split('/')) >= 3]
            if vn_index:
                if vn_index[0] == vn_index[1] == vn_index[2]:
                    vn_normal = vn_array[vn_index[0]]
                else:
                    vn_normal = [(a + b + c) / 3 for a, b, c in zip(vn_array[vn_index[0]],
                                                                    vn_array[vn_index[1]], vn_array[vn_index[2]])]
                    norm = np.linalg.norm(vn_normal)
                    vn_normal = [i / norm for i in vn_normal]
                bary_interpolator.set_face_normal(vn_normal, front_normal)
            for v_index in triangle_point_indexes:
                if v_index not in uv_map:
                    uv_map[v_index] = []
                uv_map[v_index].append(bary_interpolator)
    return uv_map


def _get_coordinate_range(v_array, col_num):
    col = [i[col_num] for i in v_array]
    return max(col) - min(col)


def _get_corner_points(v_array, principle_axes):
    points = []
    v_corner = sorted(v_array, key=lambda tup: tup[principle_axes[0]] + tup[principle_axes[1]])
    points.append(v_corner[0])
    points.append(v_corner[-1])
    v_counter_corner = sorted(v_array, key=lambda tup: tup[principle_axes[0]] - tup[principle_axes[1]])
    points.append(v_counter_corner[0])
    points.append(v_counter_corner[-1])
    return points


def _get_principle_axes(v_array):
    ranges = [_get_coordinate_range(v_array, i) for i in range(3)]
    axes = [i for i in range(3)]
    principle_axes = [i for i in axes if i != ranges.index(min(ranges))]
    non_principle_axis = [i for i in axes if i not in principle_axes]
    return principle_axes, non_principle_axis[0]


def _cache_obj(urdf_obj, obj_path):
    with open(obj_path, mode='r') as f:
        v_array, vn_array, vt_array = _retrieve_obj_elements(f)
        global_v_array = _get_global_coordinate(urdf_obj.urdf_id, v_array)
        urdf_obj.principle_axes, non_principle_axis = _get_principle_axes(global_v_array)
        urdf_obj.front_normal = AXES[non_principle_axis]
        uv_map = _get_uv_map(f, global_v_array, vt_array, vn_array, urdf_obj.front_normal)
        urdf_obj.vertices = global_v_array
        urdf_obj.uv_map = uv_map
        urdf_obj.preprocess()
        corner_points = _get_corner_points(global_v_array, urdf_obj.principle_axes)
        urdf_obj.set_start_points(corner_points)


def load_part(*args, **kwargs):
    try:
        path = args[0]
        u_id = loadURDF(*args, **kwargs)
        obj_file_path, texture_file_path = _retrieve_related_file_path(path)
        # Texture file exists, prepare for texture manipulation
        if texture_file_path:
            part = Part(u_id)
            _urdf_cache[u_id] = part
            _cache_texture(part, obj_file_path, texture_file_path)
        return u_id
    except error as e:
        print(str(e))
        return -1


def paint(urdf_id, points, color, side):
    """
    paint a specific part
    :param urdf_id: integer ID of the model returned by bullet
    :param points: intersection points in global coordinate
    :param color: list [r, g, b], each in range [0, 1]
    :param side: side of the part to be painted
    :return:
    """
    _urdf_cache[urdf_id].paint(points, color, side)


def get_job_status(urdf_id, side, color):
    return _urdf_cache[urdf_id].get_job_status(side, color)


def get_job_limit(urdf_id, side):
    return _urdf_cache[urdf_id].get_job_limit(side)


def get_texture_image(urdf_id):
    return _urdf_cache[urdf_id].get_texture_image()


def get_start_points(urdf_id, side):
    return _urdf_cache[urdf_id].get_start_points(side)


def get_guided_point(urdf_id, point, normal, delta_axis1, delta_axis2):
    return _urdf_cache[urdf_id].get_guided_point(point, normal, delta_axis1, delta_axis2)


def get_texture_size(urdf_id):
    return _urdf_cache[urdf_id].get_texture_size()
