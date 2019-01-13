import os
import math
import numpy as np
import obj_surface_process.bullet_paint_wrapper as p


def _get_target_projection_params(projection_distance):
    ratio = projection_distance / 0.5
    radius = 0.25 * ratio
    resolution = 0.02 * ratio
    target_ray_plane = projection_distance
    return radius, resolution, target_ray_plane


def _normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v


def get_pose_orn(pose, orn):
    old_z = (0, 0, 1)
    new_z = orn
    xyz = list(np.cross(old_z, new_z))
    w = float(1 + np.dot(old_z, new_z))
    xyz.append(w)
    orn = _normalize(xyz)
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
        # orn = new_orn
        orn = _normalize(orn)
        return pose, orn
    else:
        ratio_old = actual_len / target_len
        ratio_new = (target_len - actual_len) / target_len
        pose = new_pos
        orn = [a * ratio_old + b * ratio_new for a, b in zip(old_orn, new_orn)]
        # orn = new_orn
        orn = _normalize(orn)
        return pose, orn


class Robot:

    DELTA_X = 0.05
    DELTA_Y = 0.05
    PAINT_PER_ACTION = 5
    IN_POSE_TOLERANCE = 0.02
    NOT_ON_PART_TERMINATE_TIME = 1000

    def __init__(self, step_manager, urdf_path, pos=(0, 0, 0), orn=(0, 0, 0, 1), with_robot=True):
        self.robot_id = p.loadURDF(urdf_path, pos, orn, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

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
        # max velocity, etc. setup
        self._load_robot_info()
        self._with_robot = with_robot
        self._refresh_robot_pose()

        self._step_manager = step_manager
        self._reset_termination_variables()

    def _reset_termination_variables(self):
        self._terminate = False
        self._terminate_counter = 0
        self._last_on_part = True

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
            # change center of mess to wrist center.
            diff_in_end_effector = [-i for i in state[2]]
            self._pose, self._orn = _get_tcp_point_in_world(state[0], state[1], diff_in_end_effector)
        else:
            self._pose, self._orn = pos, orn

    def _generate_paint_beams(self, show_debug_lines=False):
        # Here the 0.2 could be refactored.
        radius, resolution, target_ray_plane = _get_target_projection_params(0.2)
        ray_origin = []
        ray_dst = []
        i = j = -radius
        while i <= radius:
            while j <= radius:
                # Euclidean distance within the radius
                if math.sqrt(math.pow(abs(i), 2) + math.pow(abs(j), 2)) <= radius:
                    ray_origin.append(self._pose)
                    dst_ori = [i, j, target_ray_plane]
                    dst_target, _ = _get_tcp_point_in_world(self._pose, self._orn, dst_ori)
                    ray_dst.append(dst_target)
                    if show_debug_lines:
                        p.addUserDebugLine(self._pose, dst_target, (0, 1, 0))
                j += resolution
            i += resolution
            j = -radius
        return ray_origin, ray_dst

    def _get_tcp_orn_norm(self):
        p_along_tcp, _ = p.multiplyTransforms(self._pose, self._orn, (0, 0, 1), (0, 0, 0, 1))
        vector = [b - a for a, b in zip(self._pose, p_along_tcp)]
        norm = np.linalg.norm(vector)
        norm_vector = [v / norm for v in vector]
        return norm_vector

    def _draw_tcp_orn(self):
        dst_target, _ = p.multiplyTransforms(self._pose, self._orn, (0, 0, 1), (0, 0, 0, 1))
        p.addUserDebugLine(self._pose, dst_target, (0, 1, 0))

    def _paint(self, part_id, color, paint_side, show_debug_lines=False):
        beams = self._generate_paint_beams(show_debug_lines)
        results = p.rayTestBatch(*beams)
        points = [item[3] for item in results if item[0] != -1]
        p.paint(part_id, points, color, paint_side)

    def _count_not_on_part(self):
        # check consecutive not on part
        if self._last_on_part:
            self._terminate_counter = 0
        self._terminate_counter += 1
        self._last_on_part = False
        if self._terminate_counter > Robot.NOT_ON_PART_TERMINATE_TIME:
            self._terminate = True

    def _get_actions(self, part_id, delta_axis1, delta_axis2):
        current_pose, current_orn_norm = self._pose, self._get_tcp_orn_norm()
        joint_pose = self._get_joint_pose()
        self._set_joint_pose(self._default_pos)
        act = []
        poses = {}
        delta1 = delta_axis1 / Robot.PAINT_PER_ACTION
        delta2 = delta_axis2 / Robot.PAINT_PER_ACTION
        # target_len = math.sqrt(delta1 ** 2 + delta2 ** 2)
        for i in range(Robot.PAINT_PER_ACTION):
            pos, orn_norm = p.get_guided_point(part_id, current_pose, current_orn_norm, delta1, delta2)
            # pos, orn_norm = _regularize_pose_orn(current_pose, current_orn_norm, pos, orn_norm, target_len)
            pos, orn = get_pose_orn(pos, orn_norm)
            if not pos:
                # Possible bug, along tool coordinate
                pos, _ = _get_tcp_point_in_world(current_pose, orn, [delta2, delta1, 0])
                self._count_not_on_part()
            else:
                self._last_on_part = True
            joint_angles = self._get_joint_angles(pos, orn)
            act.append(joint_angles)
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
        result = p.getJointStates(self.robot_id, self._joint_indices)
        for item in result:
            pose.append(item[0])
        return pose

    def _get_joint_angles(self, pos, orn):
        return p.calculateInverseKinematics(self.robot_id, self._end_effector_idx, pos, orn, maxNumIterations=100)

    def _check_in_position(self, pos):
        diff = [b - a for a, b in zip(pos, self._pose)]
        norm_diff = np.linalg.norm(diff)
        return True if norm_diff < Robot.IN_POSE_TOLERANCE else False

    def reset(self, pose):
        pos, orn = get_pose_orn(*pose)
        joint_angles = self._get_joint_angles(pos, orn)
        self._set_joint_pose(joint_angles)
        self._refresh_robot_pose(pos, orn)
        self._reset_termination_variables()

    def termination_request(self):
        return self._terminate

    def get_observation(self):
        return self._pose, self._get_tcp_orn_norm()

    def apply_action(self, action, part_id, color, paint_side):
        """
        support partial motor values
        :param action: tcp + normal vector
        :param color: color to be painted
        :param part_id: part id
        :param paint_side: side of the part
        :return:
        """
        for i, a in enumerate(action):
            if not -1 <= a <= 1:
                # Actually should be done by the RL framework!
                action[i] = _clip_by_value(a)
                # raise ValueError('Action {} out of range!'.format(action))
        delta_axis1 = action[0] * Robot.DELTA_X
        delta_axis2 = action[1] * Robot.DELTA_Y
        act, poses = self._get_actions(part_id, delta_axis1, delta_axis2)
        for a, pos_orn in zip(act, poses.values()):
            if self._with_robot:
                p.setJointMotorControlArray(self.robot_id, self._joint_indices, p.POSITION_CONTROL, a,
                                            forces=self._max_forces)
                # TODO: here the 100 should be refactored to a quantified criteria
                for i in range(100):
                    self._step_manager.step_simulation()
            else:
                p.addUserDebugLine(self._pose, pos_orn[0], (1, 0, 0))
            self._refresh_robot_pose(*pos_orn)
            if not self._check_in_position(pos_orn[0]):
                # Robot in singularity point or given point is out of working space
                print('not in pose!')
            self._paint(part_id, color, paint_side)
            # self._draw_tcp_orn()


if __name__ == '__main__':
    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(2.6, 180, -41, (0.0, -0.2, -0.33))
    else:
        p.connect(p.DIRECT)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    franka_urdf_path = os.path.join(current_dir, 'urdf', 'franka_description', 'robots', 'panda_arm.urdf')
    f = Robot(None, franka_urdf_path)
