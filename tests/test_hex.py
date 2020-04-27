import sys
from collections import OrderedDict
from typing import Any
from unittest.mock import Mock

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given

sys.modules["mujoco_py"] = Mock()
from opthex.hex import HexState  # isort:skip


@st.composite
def hex_state_strategy(draw: Any) -> HexState:
    """Generate a random HexState object"""
    joint_names = draw(st.lists(st.text(), unique=True))
    num_joints = len(joint_names)
    joint_pos = draw(hnp.arrays(dtype=np.float32, shape=num_joints))
    joint_vel = draw(hnp.arrays(dtype=np.float32, shape=num_joints))
    acceleration = draw(hnp.arrays(dtype=np.float32, shape=3))
    return HexState(
        qpos=OrderedDict(zip(joint_names, joint_pos)),
        qvel=OrderedDict(zip(joint_names, joint_vel)),
        acceleration=acceleration,
    )


@given(hex_state=hex_state_strategy())
def test_hex_state_joint_state(hex_state: HexState) -> None:
    """Test for converting the joint information in HexState into a np.ndarray."""
    numpy_joints = hex_state.joint_state()
    qpos = np.array(list(hex_state.qpos.values()))
    qvel = np.array(list(hex_state.qvel.values()))
    np.testing.assert_allclose(qpos, numpy_joints[0, :])
    np.testing.assert_allclose(qvel, numpy_joints[1, :])
