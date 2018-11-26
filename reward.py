import numpy as np
import collections as co
import functools as ft
import itertools as it
import operator as op
import matplotlib.pyplot as plt


def signod_barrier(x, c=0, m=1, s=1):
    expt_term = np.exp(-s * (x - c))
    result = m / (1.0 + expt_term)
    return result


def barrier_feature(s, bound=0, dim=0, sign=1, barrier_func=None):
    s_arr = np.asarray(s)
    x = (s_arr[:, dim] - bound) * sign
    return -barrier_func(x)


def barrier_punish(s, a, barrier_func=None, upper=None, lower=None):
    sn_arr = np.asarray(s)
    to_upper = sn_arr - upper
    to_lower = lower - sn_arr
    punishment = barrier_func(np.hstack([to_upper, to_lower]))
    return -1 * np.sum(punishment, axis=-1)


def l2_norm(s0, s1, rho=1):
    diff = (np.asarray(s0) - np.asarray(s1)) * rho
    return np.linalg.norm(diff)


def grid_dist(s0, s1, rho=1):
    diff = abs(s0[0] - s1[0]) + abs(s0[1] - s1[1])
    return diff * rho


def distance_punish(s, a, goal=None, dist_func=l2_norm, unit=1):
    norm = dist_func(s, goal, rho=unit)
    return -100 / (norm + 1)


def distance_reward(s, a, goal=None, dist_func=l2_norm, unit=1):
    norm = dist_func(s, goal, rho=unit)
    return 100 / (norm + 1)


def sigmoid_distance_punish(s, a, goal=None, dist_func=l2_norm, unit=1):
    norm = dist_func(s, goal, rho=unit)
    sigmod = (1.0 + np.exp(-norm))
    return -sigmod / 0.01


# def sum_rewards(s=(), a=(), func_lst=[], is_terminal=None):
#     reward = sum([f(s, a) for f in func_lst])
#     return reward

def sum_rewards(s=(), a=(), s_n=(), func_lst=[], terminals=()):
    if s in terminals:
        return 0
    reward = sum(f(s, a, s_n) for f in func_lst)
    return reward


def test_barrier_function():
    x = np.arange(-10, 10, 0.1)
    y = signod_barrier(x, m=100, s=5)
    plt.plot(x, y, 'k')
    plt.pause(0)


def test_barrier_reward():
    upper = np.array([10, 10])
    lower = np.array([0, 0])
    X = np.arange(-1, 11, 0.1)
    Y = np.arange(-1, 11, 0.1)

    XX, YY = np.meshgrid(X, Y)
    S = np.vstack([XX.flatten(), YY.flatten()]).T
    barrier_func = ft.partial(signod_barrier, c=0, m=100, s=10)
    R = barrier_punish(S, 0, barrier_func=barrier_func,
                       upper=upper, lower=lower)
    Z = R.reshape(XX.shape)
    plt.imshow(Z)
    plt.pause(0)


def test_sigmoid_distance_punish():
    goal = np.array([6, 6])
    X = np.arange(0, 11, 1)
    Y = np.arange(0, 11, 1)

    XX, YY = np.meshgrid(X, Y)
    S = np.vstack([XX.flatten(), YY.flatten()]).T

    R = sigmoid_distance(S, goal=goal, dist_func=l2_norm, unit=1)

    Z = R.reshape(XX.shape)
    plt.imshow(Z)
    plt.pause(0)


if __name__ == '__main__':
    test_sigmoid_distance_punish()
