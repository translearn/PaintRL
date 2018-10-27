import os
import pybullet as p


class Franka:

    def __init__(self, urdf_path, pos=(0, 0, 0), orn=(0, 0, 0, 1)):
        self._p = p
        self.robot_id = self._p.loadURDF(urdf_path, pos, orn, useFixedBase=True, flags=self._p.URDF_USE_SELF_COLLISION)

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
        self.joint_count = self._p.getNumJoints(self.robot_id)
        self._end_effector_idx = self.joint_count

        self._jd = [1e-5 for _ in range(self.joint_count)]
        self._max_forces = [200 for _ in range(self._motor_count)]
        joint_indices = [i for i in range(self._motor_count)]
        self._default_pos = [pos[0] for pos in self._p.getJointStates(self.robot_id, joint_indices)]

        for i in range(self.joint_count):
            joint_info = self._p.getJointInfo(self.robot_id, i)
            self._motor_lower_limits.append(joint_info[8])
            self._motor_upper_limits.append(joint_info[9])
            self._max_velocities.append(joint_info[11])

    def reset(self):
        for i in range(self._motor_count):
            self._p.resetJointState(self.robot_id, i, targetValue=self._default_pos[i])

    def get_observation(self):
        observation = []
        state = self._p.getLinkState(self.robot_id, self._end_effector_idx)
        pos, orn = state[0], state[1]
        # Why euler here?
        euler = self._p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

    def apply_action(self, motor_set_value):
        """
        support partial motor values
        :param motor_set_value: motor positions
        :return:
        """
        joint_indices = []
        for i in range(len(motor_set_value)):
            if motor_set_value not in range(self._motor_lower_limits[i], self._motor_upper_limits[i]):
                raise ValueError('The given motor value on axis {0} is out of motor limits'.format(i))
            joint_indices.append(i)

        self._p.setJointMotorControlArray(self.robot_id, joint_indices, self._p.POSITION_CONTROL, motor_set_value,
                                          force=self._max_forces)


if __name__ == '__main__':
    cid = p.connect(p.SHARED_MEMORY)
    if cid < 0:
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(2.6, 180, -41, [0.0, -0.2, -0.33])
    else:
        p.connect(p.DIRECT)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    franka_urdf_path = os.path.join(current_dir, 'urdf', 'franka_description', 'robots', 'panda_arm.urdf')
    f = Franka(franka_urdf_path)
