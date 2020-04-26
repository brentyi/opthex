import sys
from unittest.mock import Mock
sys.modules['mujoco_py'] = Mock() # mock mujoco

import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import given
from typing import Any
from collections import OrderedDict

from opthex.hex import HexState
import ipdb; ipdb.set_trace();
import mujoco_py


@st.composite
def hex_state_stragety(draw: Any) -> HexState:
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


@given(hex_state=hex_state_stragety())
def test_hex_state_joint_state(hex_state: HexState) -> None:
    """Test for converting the joint information in HexState into a np.ndarray."""
    numpy_joints = hex_state.joint_state()
    qpos = np.array(list(hex_state.qpos.values()))
    qvel = np.array(list(hex_state.qvel.values()))
    np.testing.assert_allclose(qpos, numpy_joints[0, :])
    np.testing.assert_allclose(qvel, numpy_joints[1, :])
