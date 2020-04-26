from opthex.hex import MujocoHalfHexRobot
import numpy as np

robot = MujocoHalfHexRobot()
state = robot.get_state()
joints = state.qpos

group1 = [
    "front_joint",
    "rear_joint",
]

group2 = [
    "mid_joint",
]
for joint in group1:
    joints[joint] = 0
for joint in group2:
    joints[joint] = np.pi

robot.set_joint_positions(joints)

delta = 0.004
target_states = robot.get_state().qpos
while True:
    for key in target_states.keys():
        target_states[key] += delta
    robot.set_command(target_states)
    robot.step()
