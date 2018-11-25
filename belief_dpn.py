import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools as ft
import os
import csv
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from viz import *
from reward import *
from gridworld import *


class DQNAgent:
    def __init__(self, input_shape, action_size):
        self.input_shape = input_shape
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_cnn()
        self.target_model = self._build_cnn()
        self.update_target_model()

    # def _build_model(self):
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Dense(
    #         32, input_dim=self.state_size, activation='relu'))
    #     model.add(tf.keras.layers.Dense(32, activation='relu'))
    #     model.add(tf.keras.layers.Dense(
    #         self.action_size, activation='linear'))
    #     model.compile(loss='mse',
    #                   optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
    #     return model

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _build_cnn(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        action = np.argmax(action_values[0])
        return action

    def get_state_value(self, state):
        action_values = self.model.predict(state)
        state_value = np.amax(action_values[0])
        return state_value

    def get_mean_action_values(self, state):
        action_values = self.model.predict(state)
        state_value = np.mean(action_values[0])
        return state_value

    def get_Q(self, state):
        action_values = self.model.predict(state)
        return action_values[0]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)

                states.append(state[0])
                targets.append(target[0])

        states_mb = np.array(states)
        targets_mb = np.array(targets)
        return states_mb, targets_mb

    def train(self, states_mb, targets_mb):
        history = self.model.fit(states_mb, targets_mb, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss']

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def __call__(self, state_img):
        action_values = self.model.predict(state_img)
        action_index_max = np.argmax(action_values[0])
        return action_index_max


def log_results(filename, loss_log):
    with open('results/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


def belief_reward(state, action, terminals=[]):
    agent_state, target1_state, target2_state, belief_prob = state
    r1 = distance_punish(agent_state, action, target1_state,
                         grid_dist) * belief_prob[1]
    r2 = distance_punish(agent_state, action, target2_state,
                         grid_dist) * belief_prob[2]
    reward = r1 + r2
    if agent_state in terminals:
        return -500
    return reward


if __name__ == '__main__':
    env = GridWorld("test", nx=800, ny=800)

    sheep_states = [(5, 5)]
    obstacle_states = []
    env.add_obstacles(obstacle_states)
    env.add_terminals(sheep_states)

    sheeps = {s: 500 for s in sheep_states}
    obstacles = {s: -100 for s in obstacle_states}

    S = tuple(it.product(range(env.nx), range(env.ny)))
    A = ((1, 0), (0, 1), (-1, 0), (0, -1))

    action_size = len(A)

    agent_state = ((x1, y1), (v1x1, v1x2))
    target1_state = ((x2, y2), (v2x1, v2x2))
    target2_state = ((x3, y3), (v3x1, v3x2))
    belief_prob = (p1, p2, p3)

    state = [agent_state, target1_state, target2_state, belief_prob]

    state = np.array(state)

    state_size = len(state)


    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/[(5, 5)]_episode_120.h5")
    loss_log = []

    batch_size = 32
    replay_start_size = 1000
    num_opisodes = 1001
    done = False

    for e in range(num_opisodes):
        wolf_state = random.choice(S)

        for time in range(1000):

            action = agent.act(wolf_state)
            action_grid = A[action]

            wolf_next_state = transition(
                wolf_state, action_grid, env)

            grid_reward = ft.partial(grid_reward, env=env, const=-1)
            to_sheep_reward = ft.partial(
                distance_reward, goal=sheep_states, dist_func=grid_dist, unit=1)
            func_lst = [grid_reward, to_sheep_reward]
            get_reward = ft.partial(sum_rewards, func_lst=func_lst)

            reward = get_reward(wolf_state, action)

            done = wolf_next_state in env.terminals
            next_state_img = state_to_image_array(env, image_size,
                                                  [wolf_next_state], sheeps, obstacles)
            # plt.pause(0.1)
            plt.close('all')

            next_state_img = np.reshape(
                next_state_img, [1, image_size[0], image_size[1], 3])

            agent.remember(state_img, action, reward, next_state_img, done)
            wolf_state = wolf_next_state
            state_img = next_state_img

            if len(agent.memory) > replay_start_size:
                states_mb, targets_mb = agent.replay(batch_size)
                loss = agent.train(states_mb, targets_mb)

                if time % 10 == 0:

                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                          .format(e, num_opisodes, time, loss[0]))

                    loss_log.append(loss)

            if done:
                agent.update_target_model()
                break

        if e % 10 == 0:
            module_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(module_path, "save")
            name = str(sheep_states) + '_episode_' + str(e) + '.h5'
            weight_path = os.path.join(data_path, name)
            agent.save(weight_path)

            filename = str(image_size) + '-' + \
                str(batch_size) + 'episode-' + str(e)
            log_results(filename, loss_log)
            loss_log = []
