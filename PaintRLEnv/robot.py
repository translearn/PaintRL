import os
import math
import numpy as np
from random import uniform
import bullet_paint_wrapper as p


def _random_string(length):
    import string
    import random
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))


def _get_target_projection_params(projection_distance, point_density):
    ratio = projection_distance / 0.5
    radius = 0.25 * ratio
    resolution = 1.8 / math.sqrt(point_density)
    # resolution = 0.02 * ratio * 1.414  # halved the resolution to speed up the process
    target_ray_plane = projection_distance
    return radius, resolution, target_ray_plane


def _get_uniformed_plain(point_density):
    paint_plain = []
    radius, resolution, target_ray_plane = _get_target_projection_params(0.2, point_density)
    i = j = -radius
    while i <= radius:
        while j <= radius:
            # Euclidean distance within the radius
            if math.sqrt(math.pow(i, 2) + math.pow(j, 2)) <= radius:
                paint_plain.append((i, j, target_ray_plane))
            j += resolution
        i += resolution
        j = -radius
    return paint_plain


def _get_beta_plain(beta, point_density):
    """
    beta value from paper Andulkar et al. 'Novel integrated offline trajectory generation approach
     for robot assisted spray painting operation'
    :param beta:
    :param point_density: pixel density distribution of the part,
    :return:
    """
    paint_plain = []
    distribution = {}
    radius, resolution, target_ray_plane = _get_target_projection_params(0.2, point_density)

    circles = math.ceil(radius / resolution)
    total_points = 0
    for i in range(1, circles + 1):
        num_points = (1 - (i / circles) ** 2) ** (beta - 1)
        distribution[i] = num_points
        total_points += num_points
    # TODO: here the expected points should be inferred automatically
    expected_points = 450
    for i in distribution:
        distribution[i] = round(expected_points * distribution[i] / total_points)
    for i in range(1, circles + 1):
        lower_radius = (i - 1) * resolution
        upper_radius = i * resolution
        angle_resolution = 2 * math.pi / distribution[i] if distribution[i] else 0
        for j in range(distribution[i]):
            r = uniform(lower_radius, upper_radius)
            theta = j * angle_resolution
            coordinate = pol2cart(r, theta)
            paint_plain.append((*coordinate, target_ray_plane))
    return paint_plain


def debug_plain(paint_plain):
    for point in paint_plain:
        point_positive = list(point)
        point_positive[2] = 0.1
        point_negative = list(point)
        point_negative[2] = -0.1
        p.addUserDebugLine(point_positive, point_negative, (1, 0, 0), lineWidth=2.5)


def debug_pixel():
    r = 0.1
    resolution = 0.0083333
    x = -r
    while x < r:
        y = -r
        while y < r:
            p.addUserDebugLine((x, y, -0.1), (x, y, 0.1), (0, 1, 0), lineWidth=2.5)
            y += resolution
        x += resolution


def get_pose_orn(pose, orn):
    old_z = (0, 0, 1)
    new_z = orn
    xyz = list(np.cross(old_z, new_z))
    w = float(1 + np.dot(old_z, new_z))
    xyz.append(w)
    orn = p.normalize(xyz)
    return pose, orn


def _get_tcp_point_in_world(pos, orn, point):
    return p.multiplyTransforms(pos, orn, point, (0, 0, 0, 1))


def _clip_by_value(v):
    if v < -1:
        return -1
    elif v > 1:
        return 1
    else:
        return v


def _regularize_pose_orn(old_pos, old_orn, new_pos, new_orn, target_len):
    # Limit the spatial distance of an action, in case two orientation have a big difference
    if not new_pos:
        return new_pos, new_orn
    diff = [b - a for a, b in zip(old_pos, new_pos)]
    actual_len = np.linalg.norm(diff)
    if actual_len >= target_len:
        ratio_old = target_len / actual_len
        ratio_new = (actual_len - target_len) / target_len
        diff_vec = [a * ratio_old for a in diff]
        pose = [a + b for a, b in zip(old_pos, diff_vec)]
        orn = [a * ratio_old + b * ratio_new for a, b in zip(old_orn, new_orn)]
        orn = p.normalize(orn)
        return pose, orn
    else:
        ratio_old = actual_len / target_len
        ratio_new = (target_len - actual_len) / target_len
        pose = new_pos
        orn = [a * ratio_old + b * ratio_new for a, b in zip(old_orn, new_orn)]
        orn = p.normalize(orn)
        return pose, orn


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def direction_normalize(action):
    if len(action) == 1:
        return pol2cart(1, (action[0] + 1) * np.pi)
    rho, phi = cart2pol(*action)
    x, y = abs(action[0]), abs(action[1])
    if x == 0 and y == 0:
        return x, y
    normalized_action = pol2cart(max(x, y), phi)
    # print(np.linalg.norm(normalized_action))
    return normalized_action


class Robot:

    PAINT_PER_ACTION = 5
    IN_POSE_TOLERANCE = 0.02
    NOT_ON_PART_TERMINATE_STEPS = 1000

    BETA = 2

    # Normal or fast
    PAINT_METHOD = 'fast'

    def __init__(self, step_manager, urdf_path, pos=(0, 0, 0), orn=(0, 0, 0, 1),
                 with_robot=True, capture_texture=False):

        self.robot_id = 0

        self._motor_count = 7
        self._default_pos = []
        self.joint_count = 0
        self._end_effector_idx = 0
        # joint damping coefficients
        self._jd = []
        # max forces on the 7 motors
        self._max_forces = []
        self._motor_lower_limits = []
        self._motor_upper_limits = []
        self._max_velocities = []
        self._with_robot = with_robot
        self._capture_texture = capture_texture
        # max velocity, etc. setup
        if self._with_robot:
            self.robot_id = p.loadURDF(urdf_path, pos, orn, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
            self._load_robot_info()
        self._refresh_robot_pose()

        self._step_manager = step_manager
        self._reset_termination_variables()

        self.off_part_penalty = 0
        self._last_turning_angle = 0
        self.angle_diff = 0
        self._terminate = False
        self._paint_plain = None
        self._plain_point_count = 0

    def _reset_termination_variables(self):
        self._terminate = False
        self._terminate_counter = 0
        self._last_on_part = True
        self._last_turning_angle = 0

        self.Global_path = '/tmp/' + _random_string(10) + '/'
        if self._capture_texture:
            os.mkdir(self.Global_path)
            print('Texture images will be stored under: ' + self.Global_path)
        self.Global_counter = 0

    def _load_robot_info(self):
        self.joint_count = p.getNumJoints(self.robot_id)
        self._end_effector_idx = self.joint_count - 1

        self._jd = [1e-5 for _ in range(self.joint_count)]
        self._max_forces = [200 for _ in range(self._motor_count)]
        self._joint_indices = [i for i in range(self._motor_count)]
        self._default_pos = [pos[0] for pos in p.getJointStates(self.robot_id, self._joint_indices)]

        for i in range(self.joint_count):
            joint_info = p.getJointInfo(self.robot_id, i)
            self._motor_lower_limits.append(joint_info[8])
            self._motor_upper_limits.append(joint_info[9])
            self._max_velocities.append(joint_info[11])

    def _refresh_robot_pose(self, pos=(0, 0, 0), orn=(0, 0, 0, 1)):
        if self._with_robot:
            state = p.getLinkState(self.robot_id, self._end_effector_idx)
            # change center of mess to the wrist center.
            diff_in_end_effector = [-i for i in state[2]]
            self._pose, self._orn = _get_tcp_point_in_world(state[0], state[1], diff_in_end_effector)
        else:
            self._pose, self._orn = pos, orn

    def set_up_paint_params(self, color_mode, density):
        if color_mode == 'RGB':
            self._paint_plain = _get_uniformed_plain(density)
        else:
            self._paint_plain = _get_beta_plain(self.BETA, density)
        self._plain_point_count = len(self._paint_plain)

    def _generate_paint_beams(self, show_debug_lines=False):
        ray_origin = [self._pose for _ in range(self._plain_point_count)]
        ray_dst = [_get_tcp_point_in_world(self._pose, self._orn, self._paint_plain[i])[0]
                   for i in range(self._plain_point_count)]
        if show_debug_lines:
            for point in self._paint_plain:
                p.addUserDebugLine(self._pose, point, (0, 1, 0))
        return ray_origin, ray_dst

    def _generate_paint_beam(self, show_debug_lines=False):
        ray_dst = _get_tcp_point_in_world(self._pose, self._orn, (0, 0, 0.2))[0]
        if show_debug_lines:
            self._draw_tcp_orn()
        return self._pose, ray_dst

    def _get_tcp_orn_norm(self):
        p_along_tcp, _ = p.multiplyTransforms(self._pose, self._orn, (0, 0, 1), (0, 0, 0, 1))
        vector = [b - a for a, b in zip(self._pose, p_along_tcp)]
        norm = np.linalg.norm(vector)
        norm_vector = [v / norm for v in vector]
        return norm_vector

    def _draw_tcp_orn(self):
        dst_target, _ = p.multiplyTransforms(self._pose, self._orn, (0, 0, 1), (0, 0, 0, 1))
        p.addUserDebugLine(self._pose, dst_target, (0, 1, 0))

    def _get_shot_center(self):
        return _get_tcp_point_in_world(self._pose, self._orn, (0, 0, 0.1))[0]

    def _paint(self, part_id, show_debug_lines=False):
        beams = self._generate_paint_beams(show_debug_lines)
        results = p.rayTestBatch(*beams)
        points = [item[3] for item in results if item[0] != -1]
        # return p.slow_paint(part_id, points)
        return p.paint(part_id, points, self._get_shot_center())

    def _fast_paint(self, part_id, show_debug_lines=False):
        if show_debug_lines:
            self._draw_tcp_orn()
        return p.fast_paint(part_id, self._get_shot_center())

    def _count_not_on_part(self):
        # check consecutively not on part
        if self._last_on_part:
            self._last_on_part = False
            return
        self._terminate_counter += 1
        self._last_on_part = False
        if self._terminate_counter > self.NOT_ON_PART_TERMINATE_STEPS:
            self._terminate = True

    def _get_actions(self, part_id, delta_axis1, delta_axis2):
        current_pose, current_orn_norm = self._pose, self._get_tcp_orn_norm()
        joint_pose = self._get_joint_pose()
        self._set_joint_pose(self._default_pos)
        act = []
        poses = {}
        delta1 = delta_axis1 / self.PAINT_PER_ACTION
        delta2 = delta_axis2 / self.PAINT_PER_ACTION
        # target_len = math.sqrt(delta1 ** 2 + delta2 ** 2)
        for i in range(self.PAINT_PER_ACTION):
            pos, orn_norm = p.get_guided_point(part_id, current_pose, current_orn_norm, delta1, delta2)
            # pos, orn_norm = _regularize_pose_orn(current_pose, current_orn_norm, pos, orn_norm, target_len)
            pos, orn = get_pose_orn(pos, orn_norm)
            if not pos:
                # Possible bug, along tool coordinate
                pos, _ = _get_tcp_point_in_world(current_pose, orn, [delta2, delta1, 0])
                self._count_not_on_part()
            else:
                self._last_on_part = True
            if self._with_robot:
                joint_angles = self._get_joint_angles(pos, orn)
                act.append(joint_angles)
            else:
                act.append([])
            poses[i] = [pos, orn]
            current_pose, current_orn_norm = pos, orn_norm
        self._set_joint_pose(joint_pose)
        return act, poses

    def _set_joint_pose(self, joint_angles):
        if self._with_robot:
            for i in range(self._motor_count):
                p.resetJointState(self.robot_id, i, targetValue=joint_angles[i])

    def _get_joint_pose(self):
        pose = []
        if self._with_robot:
            result = p.getJointStates(self.robot_id, self._joint_indices)
            for item in result:
                pose.append(item[0])
        return pose

    def _get_joint_angles(self, pos, orn):
        return p.calculateInverseKinematics(self.robot_id, self._end_effector_idx, pos, orn, maxNumIterations=100)

    def _check_in_position(self, pos):
        diff = [b - a for a, b in zip(pos, self._pose)]
        norm_diff = np.linalg.norm(diff)
        return True if norm_diff < self.IN_POSE_TOLERANCE else False

    def _set_turning_angle(self, delta_axis1, delta_axis2):
        if delta_axis1 != 0:
            new_angle = math.atan(abs(delta_axis2 / delta_axis1))
        else:
            new_angle = math.pi / 2
        self.angle_diff = abs(new_angle - self._last_turning_angle)
        self._last_turning_angle = new_angle

    def _simulate(self):
        # TODO: here the 100 should be refactored to a quantified criteria
        if self._step_manager.render_video:
            for i in range(100):
                self._step_manager.step_simulation()

    def reset(self, pose):
        pos, orn = get_pose_orn(*pose)
        if self._with_robot:
            joint_angles = self._get_joint_angles(pos, orn)
            self._set_joint_pose(joint_angles)
        self._refresh_robot_pose(pos, orn)
        self._reset_termination_variables()

    def get_angle_diff(self):
        return self.angle_diff

    def termination_request(self):
        return self._terminate

    def get_observation(self):
        return self._pose, self._get_tcp_orn_norm()

    def apply_action(self, action, part_id):
        """
        support partial motor values
        :param action: 1d or 2d range -1, 1
        :param part_id: part id
        :return: succeed rate of paint
        """
        for i, a in enumerate(action):
            if not -1 <= a <= 1:
                # Actually should be done by the RL framework!
                action[i] = _clip_by_value(a)
                # raise ValueError('Action {} out of range!'.format(action))
        action = direction_normalize(action)
        delta_axis1 = action[0] * p.PaintToolProfile.STEP_SIZE
        delta_axis2 = action[1] * p.PaintToolProfile.STEP_SIZE
        self._set_turning_angle(delta_axis1, delta_axis2)
        current_on_part_counter = self._terminate_counter
        act, poses = self._get_actions(part_id, delta_axis1, delta_axis2)
        possible_pixels = []
        succeeded_counter = 0
        for a, pos_orn in zip(act, poses.values()):
            if self._with_robot:
                p.setJointMotorControlArray(self.robot_id, self._joint_indices, p.POSITION_CONTROL, a,
                                            forces=self._max_forces)
            else:
                p.addUserDebugLine(self._pose, pos_orn[0], (1, 0, 0))
            self._refresh_robot_pose(*pos_orn)
            if not self._check_in_position(pos_orn[0]):
                # Robot in singularity point or given point is out of working space
                print('not in pose!')

            if self.PAINT_METHOD == 'fast':
                paint_succeed_data = self._fast_paint(part_id)
            else:
                paint_succeed_data = self._paint(part_id)

            if self._capture_texture:
                self.capture_texture_image(part_id)
            self._simulate()

            possible_pixels.extend(paint_succeed_data[0])
            succeeded_counter += paint_succeed_data[1]
        pixel_counter = len(set(possible_pixels))
        success_rate = succeeded_counter / pixel_counter if possible_pixels else 0
        if self._terminate_counter - current_on_part_counter >= self.PAINT_PER_ACTION and pixel_counter == 0:
            self.off_part_penalty = 1
            # directly terminate the process to reduce the status space
            self._terminate = True
        else:
            self.off_part_penalty = 0
        return success_rate, succeeded_counter

    def capture_texture_image(self, part_id):
        pic = p.get_texture_image(part_id)
        pic = pic.resize([480, 480])
        path = self.Global_path + str(self.Global_counter)
        print(path)
        pic.save(path + '.jpg', 'JPEG')
        self.Global_counter += 1


if __name__ == '__main__':
    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(2.6, 180, -41, (0.0, -0.2, -0.33))
    else:
        p.connect(p.DIRECT)
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF('plane.urdf', (0, 0, 0), useFixedBase=True)
    plain = _get_beta_plain(2, 14431)
    debug_plain(plain)
    debug_pixel()
