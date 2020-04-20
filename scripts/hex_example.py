from opthex.hex import MujocoHexRobot
import numpy as np

robot = MujocoHexRobot()
state = robot.get_state()
joints = state.joints

group1 = [
    "front_right_joint",
    "rear_right_joint",
    "mid_left_joint",
]

group2 = [
    "front_left_joint",
    "rear_left_joint",
    "mid_right_joint",
]
for joint in group1:
    joints[joint] = 0
for joint in group2:
    joints[joint] = np.pi

robot.set_joints(joints)

delta = 0.004
target_states = robot.get_state().joints
while True:
    for key in target_states.keys():
        target_states[key] += delta
    robot.set_command(target_states)
    robot.step()
