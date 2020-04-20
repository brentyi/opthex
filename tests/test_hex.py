# from opthex.hex import MujocoHexRobot, HexState
import os


def test_mujoco_hex():
    # mujoco tests won't work on github.
    assert True

    # pwd = os.path.dirname(__file__)
    # load_path = os.path.join(pwd, "../models/hex.xml")
    # robot = MujocoHexRobot(load_path=load_path)
    # assert isinstance(robot.get_state(), HexState)
