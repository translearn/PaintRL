import os
import math
import numpy as np
import obj_surface_process.bullet_paint_wrapper as p


def _generate_paint_beams(end_effector_pose, end_effector_orn, show_debug_lines=False):
    radius = 0.25
    resolution = 0.02
    target_ray_plane = 0.5
    ray_origin = []
    ray_dst = []
    i = j = -radius
    while i <= radius:
        while j <= radius:
            # Euclidean distance within the radius
            if math.sqrt(math.pow(abs(i), 2) + math.pow(abs(j), 2)) <= radius:
                ray_origin.append(end_effector_pose)
                dst_ori = [i, j, target_ray_plane]
                dst_target, _ = p.multiplyTransforms(end_effector_pose, end_effector_orn, dst_ori, (0, 0, 0, 1))
                ray_dst.append(dst_target)
                if show_debug_lines:
                    p.addUserDebugLine(end_effector_pose, dst_target, (0, 1, 0))
            j += resolution
        i += resolution
        j = -radius
    return ray_origin, ray_dst


def _get_tcp_orn_norm(end_effector_pose, end_effector_orn):
    p_along_tcp, _ = p.multiplyTransforms(end_effector_pose, end_effector_orn, (0, 0, 1), (0, 0, 0, 1))
    vector = [b - a for a, b in zip(end_effector_pose, p_along_tcp)]
    norm = np.linalg.norm(vector, ord=1)
    norm_vector = [v / norm for v in vector]
    return norm_vector


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


class Robot:

    DELTA_X = 0.1
    DELTA_Y = 0.1
    PAINT_PER_ACTION = 5

    def __init__(self, urdf_path, pos=(0, 0, 0), orn=(0, 0, 0, 1)):
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
        # max velocity, etc. setup.
        self._load_robot_info()

        self._pose = None
        self._orn = None
        self._refresh_robot_pose()

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

    def _refresh_robot_pose(self):
        state = p.getLinkState(self.robot_id, self._end_effector_idx)
        self._pose, self._orn = state[0], state[1]

    def _draw_tcp_orn(self):
        dst_target, _ = p.multiplyTransforms(self._pose, self._orn, (0, 0, 1), (0, 0, 0, 1))
        p.addUserDebugLine(self._pose, dst_target, (0, 1, 0))

    def _paint(self, part_id, color, show_debug_lines=False):
        beams = _generate_paint_beams(self._pose, self._orn, show_debug_lines)
        results = p.rayTestBatch(*beams)
        points = [item[3] for item in results]
        orn_norm = _get_tcp_orn_norm(self._pose, self._orn)
        p.paint(part_id, points, color, orn_norm)

    def _get_actions(self, part_id, delta_axis1, delta_axis2):
        orn_norm = _get_tcp_orn_norm(self._pose, self._orn)
        current_pose, current_orn = self._pose, orn_norm
        act = []
        delta1 = delta_axis1 / Robot.PAINT_PER_ACTION
        delta2 = delta_axis2 / Robot.PAINT_PER_ACTION
        for _ in range(Robot.PAINT_PER_ACTION):
            pos, orn_norm = p.get_guided_point(part_id, current_pose, current_orn, delta1, delta2)
            pos, orn = get_pose_orn(pos, orn_norm)
            joint_angles = p.calculateInverseKinematics(self.robot_id, self._end_effector_idx,
                                                        pos, orn, maxNumIterations=100)
            act.append(joint_angles)
            current_pose, current_orn = pos, orn_norm
        return act

    def reset(self, pose):
        pos, orn = get_pose_orn(*pose)
        joint_angles = p.calculateInverseKinematics(self.robot_id, self._end_effector_idx,
                                                    pos, orn, maxNumIterations=100)
        for i in range(self._motor_count):
            p.resetJointState(self.robot_id, i, targetValue=joint_angles[i])
        self._refresh_robot_pose()

    def get_observation(self):
        orn_norm = _get_tcp_orn_norm(self._pose, self._orn)
        return self._pose, orn_norm

    def apply_action(self, action, part_id, color):
        """
        support partial motor values
        :param action: tcp + normal vector
        :param color: color to be painted
        :param part_id: part id
        :return:
        """
        delta_axis1 = action[0] * Robot.DELTA_X
        delta_axis2 = action[1] * Robot.DELTA_Y
        act = self._get_actions(part_id, delta_axis1, delta_axis2)
        for a in act:
            p.setJointMotorControlArray(self.robot_id, self._joint_indices, p.POSITION_CONTROL, a,
                                        forces=self._max_forces)
            for i in range(100):
                p.stepSimulation()
            self._refresh_robot_pose()
            self._paint(part_id, color)
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
    f = Robot(franka_urdf_path)
