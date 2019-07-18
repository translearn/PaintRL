"""
This file wraps the urdf loading of pybullet as a new function for painting parts, adding additional
logic to process the uv mapping, please make sure the mesh obj files are processed by the process_script.py
the changeTexture function will be called implicitly as long as the new method paint is called
"""
import os
import enum
from random import randint
from pybullet import *
import xml.etree.ElementTree as Et
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree, ConvexHull, minkowski_distance

_urdf_cache = {}


def _get_texture_image(pixels, width, height):
    pixel_array = np.reshape(np.asarray(pixels), (width, height, 3))
    img = Image.fromarray(pixel_array, 'RGB')
    return img


def _clip_to_01_np(v):
    if v < 0:
        return np.float64(0)
    if v > 1:
        return np.float64(1)
    return np.float64(v)


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = np.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v


class PaintToolProfile:
    """The profile of the paint gun, static"""
    PAINT_RADIUS = 0.051
    STEP_SIZE = PAINT_RADIUS


class Side(enum.Enum):
    """First define only two sides for each part"""
    front = 1
    back = 2
    other = 3


VALID_SIDE = [Side.front, Side.back]


def _get_point_along_normal(point, length, vn):
    proportional_vn = [i * length for i in vn]
    return [a + b for a, b in zip(point, proportional_vn)]


class ConvHull:
    CORRECT_THRESHOLD = np.pi / 6

    def __init__(self, v_list, side_v_dict, front_normal, principal_axes):
        self._v_list = v_list
        self._side_v_dict = side_v_dict
        self._front_normal = front_normal
        self._principal_axes = principal_axes
        self.hull = ConvexHull(self._v_list)
        self.bary_dict = {}

    def add_debug_info(self, side):
        for bary in self.bary_dict[side]:
            bary.add_debug_info()
            bary.draw_face_normal()

    def separate_by_side(self, sides):
        for side in sides:
            self.bary_dict[side] = []
            for item in getattr(self.hull, 'simplices'):
                side_counter = 0
                for p in item:
                    if self._side_v_dict[side].data[p][0] != Part.IRRELEVANT_POSE[0]:
                        side_counter += 1
                if side_counter >= 2:
                    bary = BarycentricInterpolator(self._v_list[item[0]], self._v_list[item[1]], self._v_list[item[2]])
                    bary.align_normal()
                    bary.set_side(self._front_normal)
                    two_d_bary = bary.get_2d_bary(self._principal_axes)
                    if not bary.is_in_same_side(side):
                        bary.negate_normal()
                    self.bary_dict[side].append([bary, two_d_bary])
            # if side == Side.front:
            #     self.add_debug_info(side)

    def correct_bary_normal(self, bary):
        point = bary.center_point
        test_point = [point[self._principal_axes[0]], point[self._principal_axes[1]]]
        for three_d_bary, two_d_bary in self.bary_dict[bary.get_side()]:
            if two_d_bary.is_inside_triangle(test_point):
                if _get_included_angle(bary.get_normal(), three_d_bary.get_normal()) > self.CORRECT_THRESHOLD:
                    # self.mark_outrageous_bary(Side.front, bary, three_d_bary)
                    bary.correct_normal(three_d_bary.get_normal())
                break

    @staticmethod
    def mark_outrageous_bary(side, bary, three_d_bary):
        if bary.is_in_same_side(side):
            bary.add_debug_info()
            bary.draw_face_normal()
            three_d_bary.add_debug_info()
            three_d_bary.draw_face_normal()


class BarycentricInterpolator:
    """
    Each f line in the obj will be initialized as a barycentric interpolator
    The interpolation algorithm is taken from:
    https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
    """
    MIN_AREA = 1e-4

    def __init__(self, a, b, c):
        self._a = a
        self._b = b
        self._c = c
        self._v0 = np.subtract(b, a)
        self._v1 = np.subtract(c, a)
        self._d00 = np.dot(self._v0, self._v0)
        self._d01 = np.dot(self._v0, self._v1)
        self._d11 = np.dot(self._v1, self._v1)
        denom = (self._d00 * self._d11 - self._d01 * self._d01)
        self._inv_denom = 1.0 / denom if denom != 0 else 0

        self.area = self._get_area()
        self.area_valid = self._valid_area()

        self._uva = None
        self._uvb = None
        self._uvc = None

        self._ori_vn = None
        self._vn = None
        self._side = None

        self._get_center_point()

    def _get_center_point(self):
        center_point = []
        for a, b, c in zip(self._a, self._b, self._c):
            center_point.append((a + b + c) / 3)
        self.center_point = center_point

    def _get_bary_coordinate(self, point):
        v2 = np.subtract(point, self._a)
        d20 = np.dot(v2, self._v0)
        d21 = np.dot(v2, self._v1)
        v = (self._d11 * d20 - self._d01 * d21) * self._inv_denom
        w = (self._d00 * d21 - self._d01 * d20) * self._inv_denom
        u = 1.0 - v - w
        if self._inv_denom == 0:
            return -1, -1, -1
        return u, v, w

    @staticmethod
    def _get_pixel_coordinate(u, v, width, height):
        # pixels_coord = uv_coord * [width, height], normal round up, without Bilinear filtering
        # some uv coordinate are not in range [0, 1]
        i = min(int(round(width * u)), width - 1)
        j = min(int(round(height * v)), height - 1)
        return i, j

    def _get_area(self):
        return np.linalg.norm(np.cross(self._v0, self._v1)) / 2

    def _valid_area(self):
        return self.area >= self.MIN_AREA

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
        uva = self._get_pixel_coordinate(*self._uva, width, height)
        uvb = self._get_pixel_coordinate(*self._uvb, width, height)
        uvc = self._get_pixel_coordinate(*self._uvc, width, height)
        pixels.append(uva)
        pixels.append(uvb)
        pixels.append(uvc)
        pixel_dict = {uva: self._a, uvb: self._b, uvc: self._c}
        # Here make a 2D triangle, use barycentric coordinate to judge if pixels inside the triangle
        uv_bary = BarycentricInterpolator(self._uva, self._uvb, self._uvc)
        x_min, x_max = min(uva[0], uvb[0], uvc[0]), max(uva[0], uvb[0], uvc[0])
        y_min, y_max = min(uva[1], uvb[1], uvc[1]), max(uva[1], uvb[1], uvc[1])
        for u in range(x_min, x_max + 1):
            for v in range(y_min, y_max + 1):
                u_relative, v_relative = u / width, v / height
                if uv_bary.is_inside_triangle((u_relative, v_relative)):
                    p = uv_bary._get_bary_coordinate((u_relative, v_relative))
                    pixel_dict[(u, v)] = self.get_coordinate_in_barycentric(p)
                    pixels.append((u, v))
        return pixels, pixel_dict

    def set_side(self, front_normal):
        self._side = _get_side(self._vn, front_normal)

    def is_in_same_side(self, side):
        return self._side == side

    def get_side(self):
        return self._side

    def get_coordinate_in_barycentric(self, b_coordinate):
        bary_a, bary_b, bary_c = b_coordinate
        return tuple(np.dot(bary_a, self._a) + np.dot(bary_b, self._b) + np.dot(bary_c, self._c))

    def get_texel(self, point, width, height):
        bary_a, bary_b, bary_c = self._get_bary_coordinate(point)
        # For efficiency reason, do not check uvs are set or not
        u, v = list(np.dot(bary_a, self._uva) + np.dot(bary_b, self._uvb) + np.dot(bary_c, self._uvc))
        if u < 0 or v < 0:
            u, v = max(u, 0), max(v, 0)
        return self._get_pixel_coordinate(u, v, width, height)

    def add_debug_info(self):
        # Draw the triangle for debug
        color = (1, 0, 0)
        addUserDebugLine(self._a, self._b, color)
        addUserDebugLine(self._b, self._c, color)
        addUserDebugLine(self._c, self._a, color)

    def get_point_along_normal(self, point, length):
        return _get_point_along_normal(point, length, self._vn)

    def get_normal(self):
        return self._vn

    def get_face_guide_point(self, distance):
        target_point = self.get_point_along_normal(self.center_point, distance)
        return self.center_point, target_point

    def draw_face_normal(self):
        center_point, target_point = self.get_face_guide_point(0.15)
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

    def negate_normal(self):
        self._vn = [-i for i in self._vn]

    def correct_normal(self, vn):
        self._ori_vn = self._vn
        self._vn = vn

    def recover_normal(self):
        if self._ori_vn:
            self._vn = self._ori_vn

    def get_2d_bary(self, principal_axes):
        a_p = [self._a[principal_axes[0]], self._a[principal_axes[1]]]
        b_p = [self._b[principal_axes[0]], self._b[principal_axes[1]]]
        c_p = [self._c[principal_axes[0]], self._c[principal_axes[1]]]
        return BarycentricInterpolator(a_p, b_p, c_p)


class TextWriter:

    TEXT_COLOR = (0, 0, 0)
    TEXT_SIZE = 1.5
    LINE_SPACE = 0.07
    TOTAL_LINES = 4
    """Write episode information into the bullet environment"""

    def __init__(self, urdf_id, principal_axes, axes_ranges):
        self._urdf_id = urdf_id
        self._text_id_buffer = []
        self._principal_axes = principal_axes
        self._axes_ranges = axes_ranges
        self.lines = self.TOTAL_LINES
        self._base_pos = getBasePositionAndOrientation(urdf_id)[0]

    def _get_pos(self):
        if self.lines == 0:
            self.lines = self.TOTAL_LINES
        self.lines -= 1
        offset = list(self._base_pos)
        offset[self._principal_axes[0]] = self._axes_ranges[0][1]
        offset[self._principal_axes[1]] = self._axes_ranges[1][1]
        offset[self._principal_axes[1]] += self.lines * self.LINE_SPACE
        return offset

    def _write_line(self, line):
        text_id = addUserDebugText(line, self._get_pos(), textColorRGB=self.TEXT_COLOR,
                                   textSize=self.TEXT_SIZE)
        self._text_id_buffer.append(text_id)

    def _delete_old_info(self):
        for item_id in self._text_id_buffer:
            removeUserDebugItem(item_id)
        self._text_id_buffer.clear()

    def write_text_info(self, action, reward, penalty, total_return, step):
        self._delete_old_info()
        if isinstance(action, (int, np.int64)):
            self._write_line('Action: [{0:.3f}]'.format(round(action, 3)))
        else:
            self._write_line('Action: [{0:.3f}, {1:.3f}]'.format(round(action[0], 3), round(action[1], 3)))
        self._write_line('Reward: {0:.3f}, Penalty: {1:.3f}'.format(round(reward, 3), round(penalty, 3)))
        self._write_line('Total return: {0:.3f}'.format(round(total_return, 3)))
        self._write_line('Step: {}'.format(step))


class ColorHandler:

    def __init__(self, part):
        self._part = part
        self._color = self._part.color
        self._side = self._part.side

    def is_changed(self, texel):
        raise NotImplementedError

    def change_pixel(self, i, j):
        raise NotImplementedError

    def change_pixels(self, center, points):
        raise NotImplementedError


class RGBColorHandler(ColorHandler):

    def is_changed(self, texel):
        # Compare only the first channel to speed up the process
        return self._part.texels[texel] == self._color[0]
        # return self._part.texels[texel] == self._color[0] and self._part.texels[texel + 1] == self._color[1] \
        #        and self._part.texels[texel + 2] == self._color[2]

    def change_pixel(self, i, j):
        texel = self._part.get_texel(i, j)
        if self.is_changed(texel):
            return 0
        self._part.texels[texel] = self._color[0]
        self._part.texels[texel + 1] = self._color[1]
        self._part.texels[texel + 2] = self._color[2]
        return 1

    def change_pixels(self, _, points):
        affected_pixels = []
        succeed_counter = 0
        for index in points:
            i, j = self._part.profile[self._side][index]
            succeed_counter += self.change_pixel(i, j)
            affected_pixels.append((i, j))
        affected_pixels = list(set(affected_pixels))
        return succeed_counter, affected_pixels

    def init_part(self, color, texels):
        self._color = color
        for i, j in texels:
            self.change_pixel(i, j)
        self._color = self._part.color


class HSIColorHandler(ColorHandler):
    """
    Pseudo-HSI color handler.
    """
    TARGET_MAX = int(255 / 10)
    BETA = 2
    debug_stack = {}

    def is_changed(self, texel):
        # TODO: bugs here, need to prevent the overflow.
        return self._part.texels[texel] <= 0

    def change_pixel(self, i, j):
        """
        Here the color is used to give the quantity of changes
        :param i:
        :param j:
        :return:
        """
        texel = self._part.get_texel(i, j)
        if self.is_changed(texel):
            return 0
        self._part.texels[texel] -= self.TARGET_MAX
        self._part.texels[texel + 1] -= self.TARGET_MAX
        self._part.texels[texel + 2] -= self.TARGET_MAX
        return self.TARGET_MAX

    def _change_pixel(self, quantity, i, j):
        texel = self._part.get_texel(i, j)
        if self.is_changed(texel):
            return 0
        self._part.texels[texel] -= quantity
        self._part.texels[texel + 1] -= quantity
        self._part.texels[texel + 2] -= quantity
        return quantity / 255

    def change_pixels(self, center, points):
        x = [self._part.pixel_kd_tree[self._side].data[i] for i in points]
        y = [center] * len(x)
        distances = minkowski_distance(x, y)
        r = distances.max()
        affected_pixels = []
        succeed_counter = 0
        for counter, index in enumerate(points):
            i, j = self._part.profile[self._side][index]
            quantity = int(self.TARGET_MAX * (1 - (distances[counter] / r) ** 2) ** (self.BETA - 1)) + 1
            changed_percent = self._change_pixel(quantity, i, j)
            succeed_counter += changed_percent
            affected_pixels.append((i, j))
        affected_pixels = list(set(affected_pixels))
        return succeed_counter, affected_pixels


class Part:
    """
    Store the loaded urdf cache and its correspondent change texture parameters,
    extract the pixels on the part to be painted.
    """

    HOOK_DISTANCE_TO_PART = 0.1
    # Placeholder in the kd-tree, no point can have such a coordinate
    IRRELEVANT_POSE = (10, 10, 10)
    # Refine the feedback of the end effector pose to grid representation, prevent part overfitting
    GRID_GRANULARITY = 100
    # for the eight directions of initial painting
    MODE_SIGN = {0: [1, 0], 1: [1, -1], 2: [0, -1], 3: [-1, -1],
                 4: [-1, 0], 5: [-1, 1], 6: [0, 1], 7: [1, 1]}

    def __init__(self, urdf_id=-1, render=True, observation='section', obs_grad=10, color_mode='RGB',
                 side=Side.front, color=(1, 0, 0)):
        self.urdf_id = urdf_id
        self._render = render
        self._obs = observation
        self._obs_handler = None
        self._obs_grad = obs_grad
        self._color_mode = color_mode
        self.uv_map = None
        self.bary_list = None
        self.vertices = None
        self.vertices_kd_tree = {}
        self.texture_id = None
        self.texture_width = None
        self.texture_height = None
        self.texels = None
        self.init_texture = None
        self._start_points = {}
        self._start_pos = {}
        self.profile = {}
        self.profile_dicts = {}
        self.pixel_positions = {}
        self.pixel_kd_tree = {}
        self.principal_axes = None
        self.non_principal_axis = None
        self.ranges = None
        self.front_normal = None
        self._writer = None
        self.grid_dict = {}
        self.grid_range = {}
        self._max_grid_size = {}
        self._last_painted_pixels = []

        self._length_width_ratio = None
        # speed up the _get_texel method
        self._texel_limit = 0

        self.side = side
        self.color = self._get_color(color)
        self.color_setter = RGBColorHandler(self)
        self._color_handler = None

    @staticmethod
    def _get_color(color):
        return [np.uint8(i * 255) for i in color]

    def set_axes(self, principal_axes, non_principal_axis):
        self.principal_axes, self.non_principal_axis = principal_axes, non_principal_axis
        self.front_normal = [1 if i == self.non_principal_axis else 0 for i in range(3)]

    def set_retrieved_info(self, vertices, bary_list, uv_map):
        self.vertices, self.bary_list, self.uv_map = vertices, bary_list, uv_map

    def get_texel(self, i, j):
        return min((i + j * self.texture_width) * 3, self._texel_limit)

    def _get_closest_bary(self, point, nearest_vertex):
        closest_uvw = -1
        closest_bary = None
        for bary in self.uv_map[nearest_vertex]:
            if not bary.is_in_same_side(self.side):
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
        bary = self._get_closest_bary(point, nearest_vertex)
        if bary:
            pose = bary.get_point_along_normal(point, self.HOOK_DISTANCE_TO_PART)
            orn = [-i for i in bary.get_normal()]
            # bary.add_debug_info()
            # bary.draw_face_normal()
            return pose, orn
        return None, None

    def _refresh_texture(self):
        changeTexture(self.texture_id, self.texels, self.texture_width, self.texture_height)

    def _change_texel_color(self, bary, point):
        i, j = bary.get_texel(point, self.texture_width, self.texture_height)
        return i, j, self._color_handler.change_pixel(i, j)

    def slow_paint(self, points):
        """Only for performance comparison, do not use this method."""
        if not points:
            return [], 0
        nearest_vertices = self.vertices_kd_tree[self.side].query(points, k=1)[1]
        affected_pixels = []
        succeed_counter = 0
        for i, point in enumerate(points):
            bary = self._get_closest_bary(point, nearest_vertices[i])
            if bary:
                i, j, success = self._change_texel_color(bary, point)
                succeed_counter += success
                affected_pixels.append((i, j))
        self._refresh_texture()
        affected_pixels = list(set(affected_pixels))
        valid_pixels = [pixel for pixel in affected_pixels if pixel not in self._last_painted_pixels]
        self._last_painted_pixels = affected_pixels
        return valid_pixels, succeed_counter

    def paint(self, points, center):
        if not points:
            return [], 0
        nearest_vertices = self.pixel_kd_tree[self.side].query(points, k=1)[1]
        return self._paint(center, nearest_vertices)

    def fast_paint(self, center):
        nearest_vertices = self.pixel_kd_tree[self.side].query_ball_point(center, PaintToolProfile.PAINT_RADIUS)
        return self._paint(center, nearest_vertices)

    def _paint(self, center, nearest_vertices):
        succeed_counter, affected_pixels = self._color_handler.change_pixels(center, nearest_vertices)
        self._refresh_texture()
        valid_pixels = [pixel for pixel in affected_pixels if pixel not in self._last_painted_pixels]
        self._last_painted_pixels = affected_pixels
        return valid_pixels, succeed_counter

    def _label_part(self):
        # Preprocessing all irrelevant pixels
        target_pixels = [(i, j) for i in range(self.texture_width) for j in range(self.texture_height)]
        if self._render:
            for side in self.profile:
                target_pixels = [i for i in target_pixels if i not in self.profile[side]]
            irr_color = self._get_color((0, 0, 0))
            front_color = (0.75, 0.75, 0.75) if self._color_mode == 'RGB' else (1, 1, 1)
            front_color = self._get_color(front_color)
            back_color = self._get_color((0, 1, 0))
            # front_color = back_color = irr_color = _get_color((0.75, 0.75, 0.75))  # _get_color((1, 1, 1))
            self.color_setter.init_part(irr_color, target_pixels)
            self.color_setter.init_part(back_color, self.profile[Side.back])
            self.color_setter.init_part(front_color, self.profile[Side.front])
        # _get_texture_image(self.texels, self.texture_width, self.texture_height).show()
        else:
            color = self._get_color((0, 0, 0))
            for point in target_pixels:
                self.color_setter.change_pixel(color, *point)

    def _build_kd_tree(self):
        """
        Split the points according to the side of the part by marking the pixels
        which does not belongs to current side to IRRELEVANT_COLOR
        """
        side_label = {}
        point_list = {}
        for side in self.profile:
            point_list[side] = self.vertices.copy()

        for key, p_map in self.uv_map.items():
            for side in self.profile:
                side_label[side] = False
            for bary in p_map:
                side_label[bary.get_side()] = True
            for side, label in side_label.items():
                if not label:
                    point_list[side][key] = self.IRRELEVANT_POSE

        for side in self.profile:
            self.vertices_kd_tree[side] = cKDTree(point_list[side])
            self.pixel_kd_tree[side] = cKDTree(self.pixel_positions[side])

    def preprocess(self):
        """Store relevant pixels according to its side of the part into self.profile"""
        self._texel_limit = len(self.texels) - 4
        for bary in self.bary_list:
            if bary.get_side():
                pixels, pixel_dict = bary.get_uv_pixels(self.texture_width, self.texture_height)
                side = bary.get_side()
                if side not in self.profile:
                    self.profile[side] = []
                    self.profile_dicts[side] = {}
                    self.pixel_positions[side] = []
                self.profile[side].extend(pixels)
                self.profile_dicts[side].update(pixel_dict)
        if self.profile:
            invalid_side = None
            for side in self.profile:
                if side not in VALID_SIDE:
                    invalid_side = side
                    continue
                self.profile[side] = list(set(self.profile[side]))
                self.pixel_positions[side] = [self.profile_dicts[side][i] for i in self.profile[side]]
            del self.profile[invalid_side]
            del self.profile_dicts[invalid_side]
            del self.pixel_positions[invalid_side]
            self._label_part()
            self.init_texture = self.texels.copy()
            self._build_kd_tree()

    def _correct_bary_normals_with_conv_hull(self):
        hull = ConvHull(self.vertices, self.vertices_kd_tree, self.front_normal, self.principal_axes)
        hull.separate_by_side(self.profile.keys())
        for bary in self.bary_list:
            side = bary.get_side()
            if side == self.side:
                relative_pose = self.get_normalized_pose(bary.center_point)
                if relative_pose[0] <= 0.01 or relative_pose[0] >= 0.99 \
                        or relative_pose[1] <= 0.01 or relative_pose[1] >= 0.99:
                    continue
                hull.correct_bary_normal(bary)

    def _smooth_bary_normals_with_neighbors(self, is_debug=False):
        for side in self.profile:
            center_points = []
            for bary in self.bary_list:
                if side == bary.get_side():
                    center_points.append(bary.center_point)
                else:
                    center_points.append(self.IRRELEVANT_POSE)
            point_kd_tree = cKDTree(center_points)
            for i, bary in enumerate(self.bary_list):
                if side == bary.get_side():
                    neighbor_barys = point_kd_tree.query(bary.center_point, k=5)[1]
                    for b in neighbor_barys:
                        if self.bary_list[b] is bary:
                            continue
                        normal_angle = _get_included_angle(self.bary_list[b].get_normal(), bary.get_normal())
                        if abs(normal_angle) > np.pi / 18:
                            if is_debug:
                                self._debug_bary(b, bary)
                            else:
                                self._smooth_normal(bary, i, point_kd_tree)
                            break

    def _smooth_normal(self, bary, index, point_kd_tree):
        nearest_barys = point_kd_tree.query_ball_point(bary.center_point, PaintToolProfile.PAINT_RADIUS)
        # normals = [self.bary_list[bary_index].get_normal() for bary_index
        #            in nearest_barys if bary_index != index]
        normals = []
        for bary_index in nearest_barys:
            if bary_index != index:
                normal = self.bary_list[bary_index].get_normal()
                weighted_normal = [self.bary_list[bary_index].area * i for i in normal]
                normals.append(weighted_normal)
        if normals:
            avg_normal = np.average(normals, 0)
            correct_norm = normalize(avg_normal)
            bary.correct_normal(correct_norm)

    def _debug_bary(self, index, bary):
        bary.add_debug_info()
        bary.draw_face_normal()
        self.bary_list[index].add_debug_info()
        self.bary_list[index].draw_face_normal()

    def reset_part(self, percent, mode, with_start_point=False):
        self.texels = self.init_texture.copy()
        self._last_painted_pixels = []
        if with_start_point:
            return self.initialize_texture(percent, mode, with_start_point)
        else:
            return None

    def set_texture_info(self, t_id, width, height, texels):
        self.texture_id = t_id
        self.texture_width = width
        self.texture_height = height
        self.texels = texels

    def get_texture_size(self):
        return self.texture_width, self.texture_height

    def get_pixel_status(self, pixel):
        texel = self.get_texel(*pixel)
        return self.color_setter.is_changed(texel)

    def get_job_status(self):
        finished_counter = 0
        for pixel in self.profile[self.side]:
            if self.get_pixel_status(pixel):
                finished_counter += 1
        return finished_counter

    def get_job_limit(self):
        return len(self.profile[self.side])

    def get_texture_image(self):
        return _get_texture_image(self.texels, self.texture_width, self.texture_height)

    def set_start_points(self, corner_points):
        for side in VALID_SIDE:
            self._start_points[side] = []
        for i, point in enumerate(corner_points):
            for side in VALID_SIDE:
                pose, orn = self._get_hook_point(point, side)
                if pose:
                    self._start_points[side].append([pose, orn])

    def get_start_points(self, mode='edge'):
        """
        get the points of initial pose
        :param mode:
            fixed: only the bottom left point of the part
            anchor: only four anchor points on the four corner of a part
            edge: only valid points on the edge of the part
            all: all valid points
        :return: start points
        """
        shrink_size = PaintToolProfile.PAINT_RADIUS / 2
        start_points = []
        axis_2_value = [item[0][self.principal_axes[1]] for item in self._start_points[self.side]]
        axis_2_max, axis_2_min = max(axis_2_value), min(axis_2_value)
        for bary in self.bary_list:
            if bary.is_in_same_side(self.side) and bary.area_valid:
                center_point, hook_point = bary.get_face_guide_point(self.HOOK_DISTANCE_TO_PART)
                grid_index = self._get_grid_index_2(center_point[self.principal_axes[1]])
                grid_range = self.grid_dict[self.side][grid_index]
                if center_point[self.principal_axes[0]] - grid_range[0] >= shrink_size and \
                        grid_range[1] - center_point[self.principal_axes[0]] >= shrink_size and \
                        axis_2_min <= center_point[self.principal_axes[1]] <= axis_2_max:
                    orn = [-i for i in bary.get_normal()]
                    start_points.append([hook_point, orn])
                    # bary.add_debug_info()
                    # bary.draw_face_normal()
        if mode == 'edge':
            start_points = self._get_edge_start_points(start_points)
        if mode in ('edge', 'all'):
            self._start_points[self.side].extend(start_points)
        if mode == 'fixed':
            self._start_points[self.side] = [self._start_points[self.side][0]]
        start_pos = [item[0] for item in self._start_points[self.side]]
        self._start_pos[self.side] = cKDTree(start_pos)
        return self._start_points[self.side]

    def _get_edge_start_points(self, points):
        point_grids = {}
        start_points = []
        for point, orn in points:
            grid_index = self._get_grid_index_2(point[self.principal_axes[1]])
            if grid_index not in point_grids:
                point_grids[grid_index] = []
            point_grids[grid_index].append([point, orn])

        max_grid_index = max(point_grids.keys())
        min_grid_index = min(point_grids.keys())
        for index in point_grids:
            if index in (max_grid_index, min_grid_index):
                start_points.extend(point_grids[index])
            else:
                sorted_points = sorted(point_grids[index], key=lambda v: v[0][self.principal_axes[0]])
                # filter out the fake edge points, set the threshold to 15% the range
                x_val = sorted_points[0][0][self.principal_axes[0]]
                if (x_val - self.grid_dict[self.side][index][0]) / self.grid_range[self.side][index] < 0.15:
                    start_points.append(sorted_points[0])
                x_val = sorted_points[-1][0][self.principal_axes[0]]
                if (self.grid_dict[self.side][index][1] - x_val) / self.grid_range[self.side][index] < 0.15:
                    start_points.append(sorted_points[-1])

        return start_points

    def _correct_bary_normals(self):
        self._correct_bary_normals_with_conv_hull()
        self._smooth_bary_normals_with_neighbors()
        # self._smooth_bary_normals_with_neighbors(is_debug=True)

    def postprocess(self):
        self._length_width_ratio = (self.ranges[0][1] - self.ranges[0][0]) / (self.ranges[1][1] - self.ranges[1][0])
        self._writer = TextWriter(self.urdf_id, self.principal_axes, self.ranges)
        self._set_grid_dict()
        self._correct_bary_normals()

        if self._obs in ('section', 'discrete'):
            self._obs_handler = SectionObservation(self, self._obs_grad)
        elif self._obs == 'grid':
            self._obs_handler = GridObservation(self, self._obs_grad)
        else:
            self._obs_handler = NoObservation(self)

        if self._color_mode == 'RGB':
            self._color_handler = self.color_setter
        else:
            self._color_handler = HSIColorHandler(self)

    def get_density(self):
        area_size = 0
        step_size = (self.ranges[1][1] - self.ranges[1][0]) / self.GRID_GRANULARITY
        for grid_size_x in self.grid_range[self.side].values():
            area_size += step_size * grid_size_x
        return len(self.profile[self.side]) / area_size

    def set_ranges_along_principal(self, ranges):
        self.ranges = ranges

    def _get_grid_index_2(self, val_axis_2):
        axis2_relative = (val_axis_2 - self.ranges[1][0]) / (self.ranges[1][1] - self.ranges[1][0])
        grid_index = int(axis2_relative * self.GRID_GRANULARITY)
        if grid_index < 0:
            return 0
        elif grid_index > self.GRID_GRANULARITY - 1:
            return self.GRID_GRANULARITY - 1
        return grid_index

    def _get_delta_1(self, point, delta_axis1, delta_axis2):
        """Calculate delta 1 according to the principal 1 range size"""
        old_val_axis_2 = point[self.principal_axes[1]]
        val_axis_2 = point[self.principal_axes[1]] + delta_axis2
        old_grid_index, grid_index = self._get_grid_index_2(old_val_axis_2), self._get_grid_index_2(val_axis_2)
        if grid_index >= old_grid_index:
            grid_ranges = [self.grid_range[self.side][i] for i in range(old_grid_index, grid_index + 1)]
        else:
            grid_ranges = [self.grid_range[self.side][i] for i in range(grid_index, old_grid_index + 1)]
        avg_size = np.mean(grid_ranges)
        return delta_axis1 * avg_size / self._max_grid_size[self.side]

    def get_guided_point(self, point, normal, delta_axis1, delta_axis2):
        point = list(point)
        delta_2 = delta_axis2 * self._length_width_ratio
        # delta_1 = self._get_delta_1(point, current_side, delta_axis1, delta_2)
        delta_1 = delta_axis1
        point[self.principal_axes[0]] += delta_1
        point[self.principal_axes[1]] += delta_2
        end_point = [a + b for a, b in zip(point, normal)]
        result = rayTestBatch([point], [end_point])
        if not result[0][0] == self.urdf_id:
            if self._render:
                print('Error in Ray Test!!!')
            return None, normal
        surface_point = result[0][3]
        pos, orn = self._get_hook_point(surface_point, self.side)
        return pos, orn if orn else normal

    def initialize_texture(self, percent, mode=0, with_start_point=False):
        """
        Randomly initial the texture from 8 different sides, with different percentage
        :param with_start_point: return a start point
        :param percent: percent to be pre-painted
        :param mode: 0 = horizontal, down; 1 = right down corner; 2 = vertical, right; 3 = right up corner;
                     4 = horizontal, up; 5 = left up corner; 6 = vertical, left; 7 = left down corner.
        :return: None or start point
        """
        # self.get_texture_image().show()
        sign0, sign1 = self.MODE_SIGN[mode]
        quantity = int(len(self.profile[self.side]) * percent / 100)
        targets = sorted(self.profile[self.side], key=lambda p: sign0 * p[0] + sign1 * p[1])[:quantity]
        self.color_setter.change_pixels(None, targets)
        # self.get_texture_image().show()
        if with_start_point:
            rand_pixel = randint(quantity, len(targets) - 1)
            nearest_point = self.profile_dicts[self.side][targets[rand_pixel]]
            start_index = self._start_pos[self.side].query(nearest_point, k=1)[1]
            start_point = self._start_points[self.side][start_index]
            return start_point
        else:
            return None

    def _get_exact_boundary(self, point, is_min=True):
        proof_axis = self.principal_axes[0]
        step_size = -1e-3 if is_min else 1e-3
        steps_range = int((self.ranges[0][1] - self.ranges[0][0]) / abs(step_size))
        start_point, end_point = list(point), list(point)
        for i in range(steps_range):
            current_boundary = point[proof_axis] + i * step_size
            start_point[proof_axis] = current_boundary
            end_point[proof_axis] = current_boundary
            start_point[self.non_principal_axis] -= 1
            end_point[self.non_principal_axis] += 1
            # addUserDebugLine(start_point, end_point, (1, 0, 0))
            result = rayTestBatch([start_point], [end_point])
            if not result[0][0] == self.urdf_id:
                return current_boundary

    def _set_grid_dict(self):
        for side in self.profile:
            grid_dict = {}
            axis_1, axis_2 = self.principal_axes[0], self.principal_axes[1]
            sorted_list = sorted(self.vertices_kd_tree[side].data, key=lambda v: v[axis_2])
            sorted_list = [item for item in sorted_list if item[0] != self.IRRELEVANT_POSE[0]]
            traverse_index = 0
            axis2_range = self.ranges[1][1] - self.ranges[1][0]
            step_size = axis2_range / self.GRID_GRANULARITY
            left_bound = right_bound = sorted_list[0]
            for i in range(self.GRID_GRANULARITY):
                current_traverse_index = traverse_index
                current_step_max = self.ranges[1][0] + (i + 1) * step_size
                for index in range(current_traverse_index, len(sorted_list)):
                    if sorted_list[index][axis_2] >= current_step_max:
                        if index - current_traverse_index <= 1:
                            # use the middle value from last range
                            new_axis2 = current_step_max + 0.5 * step_size
                            if i - 1 not in grid_dict:
                                new_axis1 = sorted_list[index][axis_1]
                            else:
                                new_axis1 = (grid_dict[i - 1][0] + grid_dict[i - 1][1]) / 2
                            left_bound[axis_2] = new_axis2
                            right_bound[axis_2] = new_axis2
                            left_bound[axis_1] = new_axis1
                            right_bound[axis_1] = new_axis1
                        else:
                            target_list = sorted_list[current_traverse_index: index]
                            sorted_target_list = sorted(target_list, key=lambda x: x[axis_1])
                            left_bound = sorted_target_list[0]
                            right_bound = sorted_target_list[-1]

                        range_min = self._get_exact_boundary(left_bound)
                        range_max = self._get_exact_boundary(right_bound, is_min=False)
                        grid_dict[i] = (range_min, range_max)
                        traverse_index = index + 1
                        break
                else:
                    grid_dict[i] = (0, 0)
            self.grid_dict[side] = grid_dict
            self.grid_range[side] = {key: (value[1] - value[0]) for key, value in grid_dict.items()}
            self._max_grid_size[side] = max(self.grid_range[side].values())

    def get_normalized_pose(self, pose):
        radius = PaintToolProfile.PAINT_RADIUS
        axis1_real = pose[self.principal_axes[0]]
        axis2_real = pose[self.principal_axes[1]]
        axis2_in_range = (axis2_real - self.ranges[1][0] + radius) / (self.ranges[1][1] -
                                                                      self.ranges[1][0] + 2 * radius)
        grid_index = self._get_grid_index_2(axis2_real)
        grid_range = self.grid_dict[self.side][grid_index]
        # self._debug_grid(pose, side)
        if grid_range[1] - grid_range[0] == 0:
            axis1_in_range = 0
        else:
            axis1_in_range = (axis1_real - grid_range[0] + radius) / (grid_range[1] - grid_range[0] + 2 * radius)
        return _clip_to_01_np(axis1_in_range), _clip_to_01_np(axis2_in_range)

    def _debug_grid(self, pose):
        step_size = (self.ranges[1][1] - self.ranges[1][0]) / self.GRID_GRANULARITY
        for i in range(self.GRID_GRANULARITY):
            addUserDebugLine((pose[0], self.grid_dict[self.side][i][0], self.ranges[1][0] + (i + 1) * step_size),
                             (pose[0], self.grid_dict[self.side][i][0], self.ranges[1][0] + i * step_size), (1, 0, 0))
            addUserDebugLine((pose[0], self.grid_dict[self.side][i][1], self.ranges[1][0] + (i + 1) * step_size),
                             (pose[0], self.grid_dict[self.side][i][1], self.ranges[1][0] + i * step_size), (1, 0, 0))
            addUserDebugLine((pose[0], self.grid_dict[self.side][i][0], self.ranges[1][0] + i * step_size),
                             (pose[0], self.grid_dict[self.side][i][1], self.ranges[1][0] + i * step_size), (1, 0, 0))
            addUserDebugLine((pose[0], self.grid_dict[self.side][i][0], self.ranges[1][0] + (i + 1) * step_size),
                             (pose[0], self.grid_dict[self.side][i][1], self.ranges[1][0] + (i + 1) * step_size),
                             (1, 0, 0))

    def get_observation(self, pose):
        return self._obs_handler.get_observation(pose)

    def write_text_info(self, action, reward, penalty, total_return, step):
        self._writer.write_text_info(action, reward, penalty, total_return, step)


class Observation:

    def __init__(self, part):
        self._part = part
        self._color = self._part.color
        self._side = self._part.side

    def get_observation(self, pose):
        raise NotImplementedError


class NoObservation(Observation):

    def get_observation(self, pose):
        return None


class SectionObservation(Observation):
    # Distance weight factor, currently not supported
    DISTANCE_FACTOR = 1

    def __init__(self, part, section):
        Observation.__init__(self, part)
        self.section = section
        self._basis = 2 * np.pi / self.section

    def _get_index(self, relative_x, relative_y):
        angle = np.arctan2(relative_y, relative_x)
        if angle < 0:
            angle = 2 * np.pi + angle
        index = int(angle // self._basis)
        return index

    @staticmethod
    def _get_index_4sector(relative_x, relative_y):
        if relative_x > 0 and relative_y > 0:
            index = 0
        elif relative_x < 0 < relative_y:
            index = 1
        elif relative_x < 0 and relative_y < 0:
            index = 2
        else:
            index = 3
        return index

    def get_observation(self, pose):
        done = [0 for _ in range(self.section)]
        total = [0 for _ in range(self.section)]
        result = np.zeros(self.section, dtype=np.float64)
        for pixel, coordinate in self._part.profile_dicts[self._side].items():
            relative_x = coordinate[self._part.principal_axes[0]] - pose[self._part.principal_axes[0]]
            relative_y = coordinate[self._part.principal_axes[1]] - pose[self._part.principal_axes[1]]
            if relative_x == 0 and relative_y == 0:
                continue
            valid = 1 if not self._part.get_pixel_status(pixel) else 0
            index = self._get_index_4sector(relative_x, relative_y) if self.section == 4 \
                else self._get_index(relative_x, relative_y)
            done[index] += valid
            total[index] += 1
        for i, (d, t) in enumerate(zip(done, total)):
            result[i] = np.float64(0 if t == 0 else d / t)
        return result


class GridObservation(Observation):

    def __init__(self, part, h_grid_granularity):
        Observation.__init__(self, part)
        self._v_granularity = self._part.GRID_GRANULARITY
        self._h_granularity = h_grid_granularity
        self._setup_grid_pixels()

    def _set_pixels_in_grid(self):
        grid_pixels = {}
        axis_2_step = (self._part.ranges[1][1] - self._part.ranges[1][0]) / self._v_granularity
        for profile_dict in self._part.profile_dicts:
            grid_pixels[profile_dict] = {}
            for pixel in self._part.profile_dicts[profile_dict]:
                space_locate = self._part.profile_dicts[profile_dict][pixel]
                y_grid = min(self._v_granularity - 1,
                             int((space_locate[self._part.principal_axes[1]] - self._part.ranges[1][0]) / axis_2_step))
                if self._part.grid_range[profile_dict][y_grid] == 0:
                    x_grid = 0
                else:
                    x_step = self._part.grid_range[profile_dict][y_grid] / self._h_granularity
                    x_grid = min(self._h_granularity - 1, int((space_locate[self._part.principal_axes[0]] -
                                                               self._part.grid_dict[profile_dict][y_grid][0]) / x_step))
                if y_grid not in grid_pixels[profile_dict]:
                    grid_pixels[profile_dict][y_grid] = {}
                if x_grid not in grid_pixels[profile_dict][y_grid]:
                    grid_pixels[profile_dict][y_grid][x_grid] = []
                grid_pixels[profile_dict][y_grid][x_grid].append(pixel)
        return grid_pixels

    def _merge_vertical_grids(self, grid_pixels):
        v_interval = int(self._v_granularity / self._h_granularity)
        for profile_dict in grid_pixels:
            for v_grid in grid_pixels[profile_dict]:
                v_target = v_grid // v_interval
                for h_grid in grid_pixels[profile_dict][v_grid]:
                    pixels = grid_pixels[profile_dict][v_grid][h_grid]
                    self._grid_pixels[profile_dict][v_target][h_grid].extend(pixels)

    def _setup_grid_pixels(self):
        self._grid_pixels = {}
        for profile_dict in self._part.profile_dicts:
            self._grid_pixels[profile_dict] = {}
            for i in range(self._h_granularity):
                self._grid_pixels[profile_dict][i] = {}
                for j in range(self._h_granularity):
                    self._grid_pixels[profile_dict][i][j] = []
        grid_pixels = self._set_pixels_in_grid()
        self._merge_vertical_grids(grid_pixels)

    def _show_grids(self):
        for i in range(self._h_granularity):
            even = 1 if i % 2 == 0 else 0
            for j in range(self._h_granularity):
                if j % 2 == even:
                    continue
                num_pixels = len(self._grid_pixels[self._side][i][j])
                if num_pixels == 0:
                    continue
                for pixel in self._grid_pixels[self._side][i][j]:
                    self._part.color_setter.change_pixel(*pixel)

    def get_observation(self, _):
        # self._show_grids()
        obs = np.zeros((self._h_granularity, self._h_granularity), dtype=np.float64)
        for i in range(self._h_granularity):
            for j in range(self._h_granularity):
                num_pixels = len(self._grid_pixels[self._side][i][j])
                done_counter = 0
                if num_pixels == 0:
                    continue
                for pixel in self._grid_pixels[self._side][i][j]:
                    if self._part.get_pixel_status(pixel):
                        done_counter += 1
                obs[i][j] = np.float64(1 - done_counter / num_pixels)
        return obs.reshape((self._h_granularity ** 2,))


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


def _get_global_coordinate(urdf_id, points):
    base_pose, base_orn = getBasePositionAndOrientation(urdf_id)
    global_points = []
    for point in points:
        pose, _ = multiplyTransforms(base_pose, base_orn, point, (0, 0, 0, 1))
        global_points.append(pose)
    return global_points


def _get_coordinate_from_line(content, v_array):
    coordinate = []
    for v in content[1:]:
        coordinate.append(float(v))
    v_array.append(coordinate)


def _retrieve_obj_elements(file):
    file.seek(0)
    v_array = []
    vt_array = []
    for line in file:
        content = line.split()
        if not len(content) or content[0] == 'vn':
            continue
        if content[0] == 'v':
            _get_coordinate_from_line(content, v_array)
        elif content[0] == 'vt':
            vt_array.append([float(content[1]), 1 - float(content[2])])
    return v_array, vt_array


def _get_included_angle(a, b):
    if a == b:
        return 0
    dot_p = np.dot(a, b)
    if dot_p > 1:
        dot_p = 1
    elif dot_p < -1:
        dot_p = -1
    # assume unified vector
    return np.arccos(dot_p)


def _get_side(front_normal, v):
    max_angle_diff = np.pi / 3
    # Take care of the possible rotation made in loading the part!
    angle_front = _get_included_angle(front_normal, v)
    back_normal = [-i for i in front_normal]
    angle_back = _get_included_angle(back_normal, v)
    if - max_angle_diff <= angle_front <= max_angle_diff:
        return Side.front
    if - max_angle_diff <= angle_back <= max_angle_diff:
        return Side.back
    return Side.other


def _setup_uv_map(file, v_array, vt_array, front_normal):
    file.seek(0)
    uv_map = {}
    bary_list = []
    for line in file:
        content = line.split()
        if not len(content):
            continue
        if content[0] == 'f' and len(content) == 4:
            triangle_point_indexes = [int(i.split('/')[0]) - 1 for i in content[1:]]
            v_coordinates = [v_array[i] for i in triangle_point_indexes]
            uv_coordinates = [vt_array[int(i.split('/')[1]) - 1] for i in content[1:]]
            bary = BarycentricInterpolator(*v_coordinates)
            bary.set_uv_coordinate(*uv_coordinates)
            bary.align_normal()
            bary.set_side(front_normal)
            bary_list.append(bary)
            for v_index in triangle_point_indexes:
                if v_index not in uv_map:
                    uv_map[v_index] = []
                uv_map[v_index].append(bary)
    return bary_list, uv_map


def _get_corner_points_ranges(v_array,  principal_axes):
    shrink_size = PaintToolProfile.PAINT_RADIUS / 2
    points = []
    ranges = []
    v_corner = sorted(v_array, key=lambda v: v[principal_axes[0]] + v[principal_axes[1]])
    # Left lower corner
    v_0 = list(v_corner[0])
    v_0[principal_axes[0]] += shrink_size
    v_0[principal_axes[1]] += shrink_size
    points.append(v_0)
    # Right upper corner
    v_1 = list(v_corner[-1])
    v_1[principal_axes[0]] -= shrink_size
    v_1[principal_axes[1]] -= shrink_size
    points.append(v_1)
    v_counter_corner = sorted(v_array, key=lambda v: v[principal_axes[0]] - v[principal_axes[1]])
    # Right lower corner
    v_2 = list(v_counter_corner[0])
    v_2[principal_axes[0]] += shrink_size
    v_2[principal_axes[1]] -= shrink_size
    points.append(v_2)
    # Left upper corner
    v_3 = list(v_counter_corner[-1])
    v_3[principal_axes[0]] -= shrink_size
    v_3[principal_axes[1]] += shrink_size
    points.append(v_3)
    v_range_1 = sorted(v_array, key=lambda v: v[principal_axes[0]])
    v_range_2 = sorted(v_array, key=lambda v: v[principal_axes[1]])
    ranges.append([v_range_1[0][principal_axes[0]], v_range_1[-1][principal_axes[0]]])
    ranges.append([v_range_2[0][principal_axes[1]], v_range_2[-1][principal_axes[1]]])
    return points, ranges


def _get_coordinate_range(v_array, col_num):
    col = [i[col_num] for i in v_array]
    return max(col) - min(col)


def _get_principal_axes(v_array):
    # TODO: Refactor the principle axes to the plane perpendicular to the principal plane
    # Instead of dropping the coordinate of one direction.
    ranges = [_get_coordinate_range(v_array, i) for i in range(3)]
    non_principal_axis = ranges.index(min(ranges))
    principal_axes = [i for i in range(3) if i != non_principal_axis]
    return principal_axes, non_principal_axis


def _cache_obj(urdf_obj, obj_path):
    with open(obj_path, mode='r') as f:
        v_array, vt_array = _retrieve_obj_elements(f)
        g_v_array = _get_global_coordinate(urdf_obj.urdf_id, v_array)
        urdf_obj.set_axes(*_get_principal_axes(g_v_array))
        urdf_obj.set_retrieved_info(g_v_array, *_setup_uv_map(f, g_v_array, vt_array, urdf_obj.front_normal))
        urdf_obj.preprocess()
        corner_points, ranges = _get_corner_points_ranges(g_v_array, urdf_obj.principal_axes)
        urdf_obj.set_start_points(corner_points)
        urdf_obj.set_ranges_along_principal(ranges)
        urdf_obj.postprocess()


def _cache_texture(urdf_obj, obj_path, texture_path):
    with Image.open(texture_path) as img:
        width, height = img.size
        pixels = list(np.asarray(img.convert('RGB')).ravel())
        texture_id = loadTexture(texture_path)
        urdf_obj.set_texture_info(texture_id, width, height, pixels)
        # After this operation, the texture could be replaced after overridden the color values in pixels
        changeVisualShape(urdf_obj.urdf_id, -1, textureUniqueId=texture_id)
        _cache_obj(urdf_obj, obj_path)


def load_part(urdf_id, render, obs_mode, obs_grad, color_mode, path, side, color):
    try:
        obj_file_path, texture_file_path = _retrieve_related_file_path(path)
        if not texture_file_path:
            raise FileNotFoundError('Make sure that the .obj file is processed by Blender!')
        _urdf_cache[urdf_id] = Part(urdf_id, render, obs_mode, obs_grad, color_mode, side, color)
        _cache_texture(_urdf_cache[urdf_id], obj_file_path, texture_file_path)
    except error as e:
        print(str(e))


def paint(urdf_id, points, center):
    """
    paint a specific part
    :param urdf_id: integer ID of the model returned by bullet
    :param points: intersection points in global coordinate
    :param center: center of the paint shot
    :return: succeed data
    """
    return _urdf_cache[urdf_id].paint(points, center)


def fast_paint(urdf_id, center):
    """kd-tree method, support better pseudo-HSI color handler"""
    return _urdf_cache[urdf_id].fast_paint(center)


def slow_paint(urdf_id, points):
    """Preserved for performance comparison"""
    return _urdf_cache[urdf_id].slow_paint(points)


def get_job_status(urdf_id):
    return _urdf_cache[urdf_id].get_job_status()


def get_job_limit(urdf_id):
    return _urdf_cache[urdf_id].get_job_limit()


def get_texture_image(urdf_id):
    return _urdf_cache[urdf_id].get_texture_image()


def get_start_points(urdf_id, mode='edge'):
    return _urdf_cache[urdf_id].get_start_points(mode)


def get_guided_point(urdf_id, point, normal, delta_axis1, delta_axis2):
    return _urdf_cache[urdf_id].get_guided_point(point, normal, delta_axis1, delta_axis2)


def get_texture_size(urdf_id):
    return _urdf_cache[urdf_id].get_texture_size()


def get_normalized_pose(urdf_id, pose):
    return _urdf_cache[urdf_id].get_normalized_pose(pose)


def get_observation(urdf_id, pose):
    return _urdf_cache[urdf_id].get_observation(pose)


def reset_part(urdf_id, percent, mode, with_start_point=False):
    return _urdf_cache[urdf_id].reset_part(percent, mode, with_start_point)


def write_text_info(urdf_id, action, reward, penalty, total_return, step):
    _urdf_cache[urdf_id].write_text_info(action, reward, penalty, total_return, step)


def get_side_density(urdf_id):
    return _urdf_cache[urdf_id].get_density()
