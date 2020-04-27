import abc
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Tuple
from typing import OrderedDict as OrderedDictT

import numpy as np
from mujoco_py import MjSim, MjViewer, load_model_from_path


@dataclass(frozen=True)
class HexState:
    """Encompasses the current state of the robot.

    Joint mappings are kept in the order of the `joint_names` of the robot.
    """

    qpos: OrderedDictT[str, float]
    """A mapping between the joint name to its position."""

    qvel: OrderedDictT[str, float]
    """A mapping between the joint name to its velocity."""

    acceleration: np.ndarray
    """The acceleration vector of the robot, in the robot base frame."""
    # TODO add more ?

    def joint_state(self) -> np.ndarray:
        """A numpy representation of the joint state.

        This does not include non joint values.

        Returns
        -------
        np.ndarray, shape=(2, N)
            A numpy representation of the joint state, in the order of `joint_names`.
        """
        return np.array([list(self.qpos.values()), list(self.qvel.values())])


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
        raise NotImplementedError

    @abc.abstractmethod
    def set_command(self, command: Dict[str, float]) -> None:
        """Set the command of the robot.

        Parameters
        ----------
        command : Dict[str, float]
            The commands to the robot.
        """
        raise NotImplementedError


class BaseMujocoHexRobot(HexRobot):

    """The hexapod robot in mujoco. Child classes need to define the joint names."""

    @classmethod
    @abc.abstractproperty
    def joint_names(cls) -> Tuple[str, ...]:
        """The names of the joints in the mujoco xml."""
        raise NotImplementedError

    def __init__(self, load_path: str, viewer: bool = True):
        """Initialize a hex robot in mujoco.

        Parameters
        ---------
        load_path : str
            Path to the xml file.
        viewer : bool
            Flag to create a MjViewer for the robot. By default a viewer will be created.
        """
        model = load_model_from_path(load_path)
        self._sim = MjSim(model)

        self.num_joints = len(self.joint_names)

        self._joint_qpos_ids = [
            self._sim.model.get_joint_qpos_addr(x) for x in self.joint_names
        ]
        self._joint_qvel_ids = [
            self._sim.model.get_joint_qvel_addr(x) for x in self.joint_names
        ]
        self._joint_actuator_id_map = dict(
            zip(self.joint_names, range(self.num_joints))
        )

        if viewer:
            self._viewer = MjViewer(self._sim)
        else:
            self._viewer = None

    def get_state(self) -> HexState:
        """Retrieve the current state of the robot.

        Returns
        -------
        HexState
            The current state of the robot.
        """
        sim_state = self._sim.get_state()
        joint_pos = sim_state.qpos[self._joint_qpos_ids]
        joint_vel = sim_state.qvel[self._joint_qvel_ids]
        return HexState(
            qpos=OrderedDict(zip(self.joint_names, joint_pos)),
            qvel=OrderedDict(zip(self.joint_names, joint_vel)),
            acceleration=np.zeros(3),
        )  # TODO get the acceleration.

    def set_joint_positions(self, joints: Dict[str, float]) -> None:
        """Sets the joint positions to the specified values

        Parameters
        ----------
        joints : Dict[str, float]
            A mapping between the joint name and its value to be set.
        """
        sim_state = self._sim.get_state()
        for name, value in joints.items():
            joint_id = self._sim.model.get_joint_qpos_addr(name)
            sim_state.qpos[joint_id] = value
        self._sim.set_state(sim_state)
        self._sim.forward()

    def step(self) -> None:
        """Steps the simulation forward by one step.

        Updates the visualizer if one is available.
        """
        self._sim.step()
        if self._viewer is not None:
            self._viewer.render()

    def set_command(self, command: Dict[str, float]) -> None:
        """Set the command of the robot.

        Parameters
        ----------
        command : Dict[str, float]
            The commands to the robot.
        """
        for name, value in command.items():
            self._sim.data.ctrl[self._joint_actuator_id_map[name]] = value


class MujocoHexRobot(BaseMujocoHexRobot):

    """A hexapod robot in mujoco."""

    joint_names: Tuple[str, ...] = (
        "rear_right_joint",
        "mid_right_joint",
        "front_right_joint",
        "rear_left_joint",
        "mid_left_joint",
        "front_left_joint",
    )

    def __init__(self, load_path: str = "../models/hex.xml", viewer: bool = True):
        """Initialize a hex robot in mujoco."""
        super(MujocoHexRobot, self).__init__(load_path, viewer)


class MujocoHalfHexRobot(BaseMujocoHexRobot):

    """The half hexapod robot in mujoco. The robot only can move in the x, z plane."""

    joint_names: Tuple[str, ...] = (
        "rear_joint",
        "mid_joint",
        "front_joint",
    )

    def __init__(self, load_path: str = "../models/half_hex.xml", viewer: bool = True):
        """Initialize a half hex robot in mujoco."""
        super(MujocoHalfHexRobot, self).__init__(load_path, viewer)
