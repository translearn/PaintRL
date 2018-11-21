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


class Robot:

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

    def _load_robot_info(self):
        self.joint_count = p.getNumJoints(self.robot_id)
        self._end_effector_idx = self.joint_count - 1

        self._jd = [1e-5 for _ in range(self.joint_count)]
        self._max_forces = [200 for _ in range(self._motor_count)]
        joint_indices = [i for i in range(self._motor_count)]
        self._default_pos = [pos[0] for pos in p.getJointStates(self.robot_id, joint_indices)]

        for i in range(self.joint_count):
            joint_info = p.getJointInfo(self.robot_id, i)
            self._motor_lower_limits.append(joint_info[8])
            self._motor_upper_limits.append(joint_info[9])
            self._max_velocities.append(joint_info[11])

    def _paint(self, part_id, color, show_debug_lines=False):
        link_state = p.getLinkState(self.robot_id, self._end_effector_idx)
        beams = _generate_paint_beams(link_state[0], link_state[1], show_debug_lines)
        results = p.rayTestBatch(*beams)
        points = [item[3] for item in results]
        orn_norm = _get_tcp_orn_norm(link_state[0], link_state[1])
        p.paint(part_id, points, color, orn_norm)

    def reset(self):
        for i in range(self._motor_count):
            p.resetJointState(self.robot_id, i, targetValue=self._default_pos[i])

    def get_observation(self):
        state = p.getLinkState(self.robot_id, self._end_effector_idx)
        pos, orn = state[0], state[1]
        orn_norm = _get_tcp_orn_norm(pos, orn)
        return pos, orn_norm

    def apply_action(self, motor_set_value, part_id, color):
        """
        support partial motor values
        :param motor_set_value: motor positions
        :param color: color to be painted
        :param part_id: part id
        :return:
        """
        joint_indices = []
        for i in range(len(motor_set_value)):
            # if not self._motor_lower_limits[i] <= motor_set_value[i] <= self._motor_upper_limits[i]:
            #     raise ValueError('The given motor value on axis {0} is out of motor limits'.format(i))
            joint_indices.append(i)

        p.setJointMotorControlArray(self.robot_id, joint_indices, p.POSITION_CONTROL, motor_set_value,
                                    forces=self._max_forces)
        self._paint(part_id, color)


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
