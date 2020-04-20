import numpy as np
import os
import abc
from typing import Dict
from dataclasses import dataclass

from mujoco_py import load_model_from_path, MjSim, MjViewer


@dataclass(frozen=True)
class HexState:
    """Encompasses the current state of the robot."""

    joints: Dict[str, float]
    """Current joint state of the robot. A mapping between the joint name to its value."""

    acceleration: np.ndarray
    """The acceleration vector of the robot, in the robot frame."""
    # TODO add more


class HexRobot(abc.ABC):
    """Defines a hexapod robot."""

    @abc.abstractmethod
    def get_state(self) -> HexState:
        """Retrieve the current state of the robot.

        Returns
        -------
        RobotState
            The current state of the robot.
        """
        pass

    @abc.abstractmethod
    def set_command(self, command: Dict[str, float]) -> None:
        """Set the command of the robot.

        Parameters
        ----------
        command : Dict[str, float]
            The commands to the robot.
        """


class MujocoHexRobot(HexRobot):

    """The hexapod robot in mujoco."""

    def __init__(self, load_path: str = "../models/hex.xml", viewer: bool = True):
        """Initialize a hex robot in mujoco."""
        model = load_model_from_path(load_path)
        self.sim = MjSim(model)

        self.joint_names = [
            "rear_right_joint",
            "mid_right_joint",
            "front_right_joint",
            "rear_left_joint",
            "mid_left_joint",
            "front_left_joint",
        ]
        self.joint_ids = [
            self.sim.model.get_joint_qpos_addr(x) for x in self.joint_names
        ]
        self.joint_name_id_map = dict(zip(self.joint_names, self.joint_ids))
        self.joint_actuator_id_map = dict(zip(self.joint_names, list(range(6))))

        if viewer:
            self.viewer = MjViewer(self.sim)
        else:
            self.viewer = None

    def get_state(self) -> HexState:
        """Retrieve the current state of the robot.

        Returns
        -------
        HexState
            The current state of the robot.
        """
        sim_state = self.sim.get_state()
        joints = sim_state.qpos[self.joint_ids]
        return HexState(
            joints=dict(zip(self.joint_names, joints)), acceleration=0
        )  # TODO get the acceleration.

    def set_joints(self, joints: Dict[str, float]) -> None:
        """Sets the joints to the specified values

        Parameters
        ----------
        joints : Dict[str, float]
            A mapping between the joint name and its value to be set.
        """
        sim_state = self.sim.get_state()
        for name, value in joints.items():
            joint_id = self.joint_name_id_map[name]
            sim_state.qpos[joint_id] = value
        self.sim.set_state(sim_state)
        self.sim.forward()

    def step(self) -> None:
        """Steps the simulation forward by one step. Updates the visualizer if one is available."""
        self.sim.step()
        if self.viewer is not None:
            self.viewer.render()

    def set_command(self, command: Dict[str, float]) -> None:
        """Set the command of the robot.

        Parameters
        ----------
        command : Dict[str, float]
            The commands to the robot.
        """
        ctrl = np.zeros((6,))
        for name, value in command.items():
            ctrl[self.joint_actuator_id_map[name]] = value
        self.sim.data.ctrl[:] = ctrl
