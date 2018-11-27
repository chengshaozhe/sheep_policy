import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools as ft
import os
import csv
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from viz import *
from reward import *
from gridworld import *
from BeliefUpdate import *
from PreparePolicy import *
import Transition

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._buildRNN()
        self.target_model = self._buildRNN()
        self.updateTargetModel()

    # def _buildDNN(self):
    #     model = tf.keras.Sequential()
    #     model.add(tf.keras.layers.Dense(
    #         32, input_dim=self.state_size, activation='relu'))
    #     model.add(tf.keras.layers.Dense(32, activation='relu'))
    #     model.add(tf.keras.layers.Dense(
    #         self.action_size, activation='linear'))
    #     model.compile(loss='mse',
    #                   optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
    #     return model

    def _huberLoss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * \
            K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _buildDNN(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _buildRNN(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(1,self.state_size), return_sequences=False))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        action = np.argmax(action_values[0])
        return action

    def getStateValue(self, state):
        action_values = self.model.predict(state)
        state_value = np.amax(action_values[0])
        return state_value

    def getMeanActionValues(self, state):
        action_values = self.model.predict(state)
        state_value = np.mean(action_values[0])
        return state_value

    def getQ(self, state):
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


def logResults(filename, loss_log):
    with open('results/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


def isTerminals(state):
    agent_state = state[:5]
    wolf_state = state[5:10]
    distractor_state = state[10:15]

    agent_coordinates = agent_state[:2]
    wolf_coordinates = wolf_state[:2]

    if np.around(agent_coordinates).all == np.around(wolf_coordinates).all:
        return True
    return False

def beliefReward(state, action):
    agent_state = state[:5]
    wolf_state = state[5:10]
    distractor_state = state[10:15]

    d1 = distance_punish(agent_state[:2], action, wolf_state[:2], grid_dist) 
    d2 = distance_punish(agent_state[:2], action, distractor_state[:2], grid_dist) 

    p1 = np.exp(wolf_state[-1])
    p2 = np.exp(distractor_state[-1])

    reward = d1 * p1 + d2 * p2

    if isTerminals(state):
        return -500
    return reward


if __name__ == '__main__':
    # env = GridWorld("test", nx=800, ny=800)
    # obstacle_states = []
    # env.add_obstacles(obstacle_states)
    # obstacles = {s: -100 for s in obstacle_states}

    # agent_state = [x1, y1, v1x1, v1x2]
    # wolf_state = [x2, y2, v2x1, v2x2]
    # distractor_state = [x3, y3, v3x1, v3x2]
    # belief_prob = [p1, p2, p3]
    # state = [agent_state, wolf_state, distractor_state, belief_prob]

    statesList = [[10,10,0,0],[10,5,0,0],[15,15,0,0]]
    speedList = [5,3,3]
    movingRange=[0,0,15,15]
    assumeWolfPrecisionList=[50,1.3]
    sheepIdentity=0
    wolfIdentity=1
    distractorIdentity=2
    wolfPrecision=50
    distractorPrecision=0.5/3.14
    
    transState = Transition.Transition(movingRange, speedList)
    updateBelief = BeliefUpdate(assumeWolfPrecisionList, sheepIdentity)
    takeWolfAction = WolfPolicy(sheepIdentity, wolfIdentity, speedList[wolfIdentity])
    takeDistractorAction = DistractorPolicy(distractorIdentity, distractorPrecision, speedList[distractorIdentity])

    numOfActions = 16
    actionAnglesList = [i * (360 / numOfActions)
                        for i in range(1, numOfActions + 1)]
    sheepActionList = [(speedList[sheepIdentity] * np.cos(actionAngles * np.pi / 180),
                    speedList[sheepIdentity] * np.sin(actionAngles * np.pi / 180)) for actionAngles in actionAnglesList]

    state_size = 15
    action_size = numOfActions
    agent = DQNAgent(state_size, action_size)
    # agent.load("./save/[(5, 5)]_episode_120.h5")
    loss_log = []

    batch_size = 16
    replay_start_size = 21
    num_opisodes = 1001
    done = False

    for e in range(num_opisodes):

        statesList = [[random.randint(0,800),random.randint(0,800),0,0],
                        [random.randint(0,800),random.randint(0,800),0,0],
                        [random.randint(0,800),random.randint(0,800),0,0]]

        oldStates = pd.DataFrame(statesList,index=[0,1,2],columns=['positionX','positionY','velocityX','velocityY'])
        oldBelief = initiateBeliefDF(statesList)

        done = False

        for time in range(1000):

            oldBelief_input = np.asarray(oldBelief).flatten()
            # oldBelief_input = np.reshape(oldBelief_input,[1,state_size])
            oldBelief_input = np.reshape(oldBelief_input,[1,1,state_size]) # LSTM 

            action = agent.act(oldBelief_input)

            sheepAction = sheepActionList[action]
            # print (action, sheepAction)

            wolfAction = takeWolfAction(oldStates, wolfPrecision)

            distractorAction = takeDistractorAction(oldStates)

            currentActions = [sheepAction, wolfAction, distractorAction]
            # print (currentActions)
            
            currentStates = transState(oldStates, currentActions)
            currentBelief = updateBelief(oldBelief, currentStates)

            currentBelief_input = np.asarray(currentBelief).flatten()
            reward = beliefReward(currentBelief_input, currentActions)

            # print(currentBelief_input)
            # print (reward)

            if isTerminals(currentBelief_input):
                done = 1
            else:
                done = 0

            # plt.pause(0.01)
            # plt.close('all')

            currentBelief_input = np.reshape(currentBelief_input,[1,1,state_size]) # LSTM input
            # currentBelief_input = np.reshape(currentBelief_input,[1,state_size])
            agent.remember(oldBelief_input, action, reward, currentBelief_input, done)

            oldStates = currentStates
            oldBelief = currentBelief

            if len(agent.memory) > replay_start_size:
                states_mb, targets_mb = agent.replay(batch_size)
                loss = agent.train(states_mb, targets_mb)

                if time % 10 == 0:

                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                          .format(e, num_opisodes, time, loss[0]))

                    loss_log.append(loss)

            if done:
                agent.updateTargetModel()
                break

        if e % 10 == 0:
            module_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(module_path, "save")
            name = 'episode_' + str(e) + '.h5'
            weight_path = os.path.join(data_path, name)
            agent.save(weight_path)

            filename = str(action_size) + '-' + \
                str(batch_size) + 'episode-' + str(e)
            logResults(filename, loss_log)
            loss_log = []
