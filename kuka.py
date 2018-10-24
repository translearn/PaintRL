"""
taken from example
"""
import pybullet as p
import pybullet_data
import math
from datetime import datetime


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)
kukaId = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
joints = []
velocities = []
forces = []
pos_gains = []
vel_gains = []
for i in range(numJoints):
    joints.append(i)
    forces.append(500)
    velocities.append(0)
    pos_gains.append(0.03)
    vel_gains.append(1)
if numJoints != 7:
    exit()

# restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
# joint damping coefficients
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

for i in range(numJoints):
    p.resetJointState(kukaId, i, rp[i])

p.setGravity(0, 0, -10)
t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1

useOrientation = 1
useSimulation = 1
useRealTimeSimulation = 1
p.setRealTimeSimulation(useRealTimeSimulation)

trailDuration = 15


while True:
    dt = datetime.now()
    t = (dt.second / 60.) * 2. * math.pi

    pos = [-0.4, 0.2 * math.cos(t), 0. + 0.2 * math.sin(t)]
    # end effector points down, not up (in case useOrientation==1)
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])
    jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, jointDamping=jd)
    p.setJointMotorControlArray(bodyIndex=kukaId, jointIndices=joints, controlMode=p.POSITION_CONTROL,
                                targetPositions=jointPoses, targetVelocities=velocities,
                                forces=forces, positionGains=pos_gains, velocityGains=vel_gains)

    ls = p.getLinkState(kukaId, kukaEndEffectorIndex)
    if hasPrevPose:
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], 1, trailDuration)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], 1, trailDuration)
    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1
