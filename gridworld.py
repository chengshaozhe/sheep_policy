
import numpy as np
from viz import *
from reward import *
import random
from collections import deque
import os
from PIL import Image


class GridWorld():
    def __init__(self, name='', nx=None, ny=None):
        self.name = name
        self.nx = nx
        self.ny = ny
        self.coordinates = tuple(it.product(range(self.nx), range(self.ny)))
        self.terminals = []
        self.obstacles = []
        self.features = co.OrderedDict()

    def add_terminals(self, terminals=[]):
        for t in terminals:
            self.terminals.append(t)

    def add_obstacles(self, obstacles=[]):
        for o in obstacles:
            self.obstacles.append(o)

    def add_feature_map(self, name, state_values, default=0):
        self.features[name] = {s: default for s in self.coordinates}
        self.features[name].update(state_values)

    def is_state_valid(self, state):
        if state[0] not in range(self.nx):
            return False
        if state[1] not in range(self.ny):
            return False
        if state in self.obstacles:
            return False
        return True

    def reward(self, s, a, s_n, W={}):
        if not W:
            return sum(map(lambda f: self.features[f][s_n], self.features))
        return sum(map(lambda f: self.features[f][s_n] * W[f], W.keys()))

    def draw_feature(self, ax, name, **kwargs):
        I = dict_to_array(self.features[name])
        return draw_2D_array(I, ax, **kwargs)

    def draw_features_first_time(self, ax, features=[], colors={},
                                 masked_values={}, default_masked=0):
        assert set(features).issubset(set(self.features.keys()))

        if not features:
            features = self.features.keys()
        if len(features) > len(color_set):
            raise ValueError("there are %d features and only %d colors"
                             % (len(features), len(color_set)))

        free_color = list(filter(lambda c: c not in colors.values(),
                                 color_set))
        colors.update({f: free_color.pop(0)
                       for f in features if f not in colors.keys()})
        masked_values.update({f: default_masked
                              for f in features if f not in masked_values.keys()})

        assert set(masked_values.keys()) == set(colors.keys()) == set(features)

        if not ax:
            fig, ax = plt.subplots(1, 1, tight_layout=True)

        def single_feature(ax, name):
            f_color = colors[name]
            masked_value = masked_values[name]

            return self.draw_feature(ax, name, f_color=f_color,
                                     masked_value=masked_value)

        ax_images = {f: single_feature(ax, f) for f in features}
        return ax, ax_images

    def update_features_images(self, ax_images, features=[], masked_values={},
                               default_masked=0):
        def update_single_feature(name):
            try:
                masked_value = masked_values[name]
            except:
                masked_value = default_masked
            I = dict_to_array(self.features[name])
            return update_axes_image(ax_images[name], I, masked_value)
        return {f: update_single_feature(f) for f in features}

    def draw(self, ax=None, ax_images={}, features=[], colors={},
             masked_values={}, default_masked=0, show=False, save_to=''):

        plt.cla()
        if ax:
            ax.get_figure()
        new_features = [f for f in features if f not in ax_images.keys()]
        old_features = [f for f in features if f in ax_images.keys()]
        ax, new_ax_images = self.draw_features_first_time(ax, new_features,
                                                          colors, masked_values, default_masked=0)
        old_ax_images = self.update_features_images(ax_images, old_features,
                                                    masked_values,
                                                    default_masked=0)
        ax_images.update(old_ax_images)
        ax_images.update(new_ax_images)

        # if save_to:
        #     fig_name = os.path.join(save_to, str(self.name) + ".png")
        #     plt.savefig(fig_name, dpi=200)
        #     if self.verbose > 0:
        #         print ("saved %s" % fig_name)
        # if show:
        #     plt.show()

        return ax, ax_images

# def reward(s, a, env=None, const=-10, is_terminal=None):
#     return const + sum(map(lambda f: env.features[f][s], env.features))


def grid_reward(s, a, sn, env=None, const=-1):
    goal_reward = env.features['sheep'][sn] if sn in env.terminals else const
    obstacle_punish = env.features['obstacle'][sn] if sn in env.obstacles else 0
    return goal_reward + obstacle_punish


def physics(s, a, env=None):
    if s in env.terminals:
        return s
    s_n = tuple(map(sum, zip(s, a)))
    if env.is_state_valid(s_n):
        return s_n
    return s


def getValidActions(s, A, env=None):
    valid_actions = []
    for a in A:
        s_n = tuple(map(sum, zip(s, a)))
        if env.is_state_valid(s_n):
            valid_actions.append[a]
    return valid_actions


def state_to_image_array(env, image_size, wolf_states, sheeps, obstacles):
    wolf = {s: 1 for s in wolf_states}
    env.add_feature_map("wolf", wolf, default=0)
    env.add_feature_map("sheep", sheeps, default=0)
    env.add_feature_map("obstacle", obstacles, default=0)

    ax, _ = env.draw(features=("wolf", "sheep", "obstacle"), colors={
                     'wolf': 'r', 'sheep': 'g', 'obstacle': 'y'})

    fig = ax.get_figure()
    # fig.set_size_inches((image_size[0] / fig.dpi, image_size[1] / fig.dpi)) # direct resize
    fig.canvas.draw()

    image = np.fromstring(fig.canvas.tostring_rgb(),
                          dtype=np.uint8, sep='')
    image_array = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# use PIL to resize
    pil_im = Image.fromarray(image_array)
    image_array = np.array(pil_im.resize(image_size[:2], 3))

    # print (image_array.shape)
    # print (len(np.unique(image_array)))
    return image_array
