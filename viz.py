# for linux ssh: set matplotlib to not use the Xwindows backend.
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# plt.ion()
# plt.ioff()

import matplotlib as mp
import numpy as np

color_set = ('g', 'r', 'c', 'm', 'y', 'k', 'w', 'b')
# color_set = ('r', 'g', 'b', 'y', 'm', 'c', 'w', 'b')


def create_color_map(input_colors, bounds=(0, 0.5, 1), bad_color='white'):
    assert set(input_colors) < set(color_set)
    cmap = mp.colors.ListedColormap(input_colors)
    norm = mp.colors.BoundaryNorm(bounds, cmap.N)
    cmap.set_bad(color=bad_color, alpha=0)
    cmap.set_bad
    return cmap, norm


def draw_2D_array(I, ax=None, b_color='b', f_color='r', bounds=(), g_color='k',
                  masked_value=None, **kwargs):
    """draw an 2D array on a gr ax"""
    assert f_color in color_set
    assert b_color in color_set
    if not ax:
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        # ax.yaxis.tick_right()  # remove right y-Ticks

    for v in np.arange(-0.5, I.shape[1] + 0.5, 1):
        ax.axvline(v, lw=1, color='k', zorder=10)
    for h in np.arange(-0.5, I.shape[0] + 0.5, 1):
        ax.axhline(h, lw=1, color='k', zorder=10)

    # get the bounds of the array
    if not bounds:
        l = np.min(I)
        h = np.max(I)
        bounds = (l, (l + h) / 2, h)
    # get the masked array
    if masked_value is not None:
        I = np.ma.masked_values(I, masked_value)
    cmap, norm = create_color_map((b_color, f_color), bounds=bounds)
    ax_image = ax.axes.matshow(I, cmap=cmap, norm=norm, **kwargs)
    return ax_image


def V_array_to_dict(V_arr, S=()):
    return {s: V_arr[si] for (si, s) in enumerate(S)}


def dict_to_array(V):
    states, values = zip(*((s, v) for (s, v) in V.items()))
    row_index, col_index = zip(*states)
    num_row = max(row_index) + 1
    num_col = max(col_index) + 1
    I = np.empty((num_row, num_col))
    I[row_index, col_index] = values
    return I


def draw_V(ax=None, V=None, S=()):
    if isinstance(V, dict):
        V_dict = V
    elif isinstance(V, np.ndarray):
        V_dict = V_array_to_dict(V, S=S)
    data = dict_to_array(V_dict)

    ax.axes.matshow(data)
    for v in np.arange(-0.5, data.shape[1] + 0.5, 1):
        ax.axvline(v, lw=1, color='k', zorder=10)
    for h in np.arange(-0.5, data.shape[0] + 0.5, 1):
        ax.axhline(h, lw=1, color='k', zorder=10)

    ax.set_xticks(np.arange(0, data.shape[0], 1))
    ax.set_yticks(np.arange(0, data.shape[1], 1))

    return ax


def draw_state_quiver(ax, state=(), actions={}, scale=1.5, color='k', **kwargs):
    X, Y, U, V = zip(*((state[1], state[0], a[1] * v, a[0] * v)
                       for (a, v) in actions.iteritems()))

    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scale,
              color=color, **kwargs)

    return ax


def draw_4D_V(axes=None, V=None, s1=(), s2=(), s3=(), s4=(), pause=0, **kwargs):
    n1, n2, n3, n4 = len(s1), len(s2), len(s3), len(s4)
    if hasattr(axes, "__len__"):
        assert axes.shape == (n3, n4)  # draw d3 and d4 in subplots
        fig = axes.flatten()[0].get_figure()
    else:
        fig, axes = plt.subplots(n3, n4)
    V_plots = V.reshape(n1 * n2, n3 * n4)

    vmax = np.max(V)
    vmin = np.min(V)

    for (i, sa) in enumerate(it.product(s3, s4)):
        ax = axes.flatten()[i]
        V_a = V_plots[:, i].reshape(n1, n2)
        mid_x = V_a.shape[0] // 2
        mid_y = V_a.shape[1] // 2
        draw_V_arr(ax=ax, V=V_a, pause=0, vmin=vmin, vmax=vmax, **kwargs)
        draw_state_quiver(ax, (mid_x, mid_y), {sa: 1}, scale=2 * 1.0 / mid_x)
    cax = ax.images[0]
    fig.colorbar(cax, ticks=np.linspace(vmin, vmax, 5), orientation='vertical')
    if pause:
        plt.pause(pause)
    return axes


def draw_T(axes=None, T=None, S=(), a=(), ai=None):
    axes_f = axes.flatten()
    for si, s in enumerate(S):
        ax = axes_f[si]
        pT = T[si, ai, :]
        draw_V(ax=ax, V=pT, S=S)
        draw_state_quiver(ax, s, {a: 1}, scale=2, color='w')


def draw_R(axes=None, R=None, S=(), a=(), ai=None):
    axes_f = axes.flatten()
    for si, s in enumerate(S):
        ax = axes_f[si]
        pT = R[si, ai, :]
        draw_V(ax=ax, V=pT, S=S)
        draw_state_quiver(ax, s, {a: 1}, scale=2, color='w')


def draw_Q(axes=None, V=None, S=(), A=()):
    axes_f = axes.flatten()
    for si, s in enumerate(S):
        ax = axes_f[si]
        pT = R[si, ai, :]
        draw_V(ax=ax, V=pT, S=S)
        draw_state_quiver(ax, s, {a: 1}, scale=2, color='w')


def draw_policy(ax=None, PI={}, V=None, S=()):
    # axes_f = ax.flatten()
    draw_V(ax, V, S)
    for s in S:
        draw_state_quiver(ax, s, {PI[s]: 1}, scale=2, color='w')


# def draw_episode(ax=None, start=(), t=0.1, env=None, pi_func=None, physics=None,
#                  is_terminal=None, max_iter=100, pause=0, verbose=1):
#     state = start
#     env.draw(ax)
#     for i in range(max_iter):
#         action = pi_func(state)
#         draw_state_quiver(ax, state, {action:1}, scale=0.75/t)
#         state = physics(state, action, t=t)
#         if is_terminal(state):
#             if verbose>0:
#                 print ("reaching a terminal state")
#             break
#         if i==max_iter-1:
#             if verbose>0:
#                 print ("reaching maximum episode length ")
#             break
#     if pause:
#         plt.pause(pause)
#     return ax

def update_axes_image(ax_image, I, masked_value=0):
    I_masked = np.ma.masked_equal(I, masked_value)
    ax_image.set_array(I_masked)
    ax_image.axes.get_figure().canvas.draw()
    return ax_image


def draw_policy_4d(ax=None, PI={}, V=None, S=()):
    draw_V_4d(ax, V, S)
    for s in S:
        draw_state_quiver(ax, s, {PI[s]: 1}, scale=2, color='w')


def draw_policy_4d_softmax(ax=None, PI={}, V=None, S=(), A=()):
    draw_V(ax, V, S)
    for s in S:
        for a in A:
            draw_state_quiver_softmax(ax, s, PI[s], scale=2, color='w')


def draw_state_quiver_softmax(ax, state=(), actions={}, scale=1.5, color='k', **kwargs):
    X, Y, U, V = zip(*((state[1], state[0], a[1] * v, a[0] * v)
                       for (a, v) in actions.iteritems()))

    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=scale,
              color=color, **kwargs)

    return ax


def draw_V_4d(ax=None, V=None, S=()):
    if isinstance(V, dict):
        V_dict = V
    elif isinstance(V, np.ndarray):
        V_dict = V_array_to_dict(V, S=S)
    data = dict_to_array_4d(V_dict)

    ax.axes.matshow(data)
    for v in np.arange(-0.5, data.shape[1] + 0.5, 1):
        ax.axvline(v, lw=1, color='k', zorder=10)
    for h in np.arange(-0.5, data.shape[0] + 0.5, 1):
        ax.axhline(h, lw=1, color='k', zorder=10)

    ax.set_xticks(np.arange(0, data.shape[0], 1))
    ax.set_yticks(np.arange(0, data.shape[1], 1))

    return ax


def draw_episode(s, ax=None, gmap=None, pause=0, scale=1.5):
    if not ax:
        ax, _ = env.draw()
    draw_state_quiver(ax, s[:2], {action: 1}, scale=scale)
    if pause:
        plt.pause(pause)
    return ax


def dict_to_array_4d(V):
    states, values = zip(*((s, v) for (s, v) in V.iteritems()))
    row_index, col_index = zip(*states)
    num_row = max(row_index) + 1
    num_col = max(col_index) + 1
    I = np.empty((num_row, num_col))
    I[row_index, col_index] = values
    return I
