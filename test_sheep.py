import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools as ft
import os
import csv
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K
from time import time

from viz import *
from reward import *
from gridworld import *
from BeliefUpdate import *
from PreparePolicy import *
from InitialPosition import *
import Attention
import Transition

from BeliefAttentionDQN import *

import unittest
from ddt import ddt, data, unpack


def barrier_punish(pos, movingRange):
    sn_arr = np.asarray(pos)

    lower = np.asarray(movingRange[:2])
    upper = np.asarray(movingRange[2:])
    margin = np.array([0, 0])

    to_upper = upper - sn_arr + margin
    to_lower = sn_arr - lower - margin

    punishment = signod_barrier(
        np.hstack([to_upper, to_lower]), m=15, s=1 / 1000)
    # print(np.sum(punishment, axis=-1))
    reward = min(np.sum(punishment, axis=-1) - 54, 1)
    return reward


@ddt
class TestMDP(unittest.TestCase):
    @data((10, 10), (20, 20), (30, 30), (40, 40), (50, 50), (60, 60), (180, 180), (240, 240))
    @unpack
    def test_beliefReward(self, state, index):
        movingRange = [0, 0, 364, 364]
        action = (0, 0)
        self.assertEqual(barrier_punish(state, movingRange),
                         0)


if __name__ == '__main__':

    unittest.main()
