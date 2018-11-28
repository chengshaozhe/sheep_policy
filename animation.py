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

    if l2_norm(agent_coordinates, wolf_coordinates) <= 10:
    # if np.around(agent_coordinates).all == np.around(wolf_coordinates).all:
        return True
    return False

def beliefReward(state, action):
    agent_state = state[:5]
    wolf_state = state[5:10]
    distractor_state = state[10:15]

    d1 = l2_norm(agent_state[:2], wolf_state[:2])
    d2 = l2_norm(agent_state[:2], distractor_state[:2])

    p1 = wolf_state[-1]
    p2 = distractor_state[-1]

    const = 3
    reward = -1000/d1 * p1 - 1000/d2 * p2

    if isTerminals(state):
        return -500
    return reward + const


if __name__ == '__main__':
    statesListInit = [[10,10,0,0],[10,5,0,0],[15,15,0,0]]
    speedList = [10,8,8]
    movingRange=[0,0,800,800]
    assumeWolfPrecisionList = [50,11,3.3,1.83,0.92,0.31]
    sheepIdentity=0
    wolfIdentity=1
    distractorIdentity=2
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

        statesList = [[random.randint(0,movingRange[2]),random.randint(0,movingRange[3]),0,0],
                        [random.randint(0,movingRange[2]),random.randint(0,movingRange[3]),0,0],
                        [random.randint(0,movingRange[2]),random.randint(0,movingRange[3]),0,0]]

        oldStates = pd.DataFrame(statesList,index=[0,1,2],columns=['positionX','positionY','velocityX','velocityY'])
        oldBelief = initiateBeliefDF(statesList)

        wolfPrecision = random.choice(assumeWolfPrecisionList)
        done = False



        for time in range(1000):

            oldBelief_input = np.asarray(oldBelief).flatten()
            # oldBelief_input = np.reshape(oldBelief_input,[1,state_size])
            oldBelief_input = np.reshape(oldBelief_input,[1,1,state_size]) # LSTM 
            # print(oldBelief_input)

            action = agent.act(oldBelief_input)

            sheepAction = sheepActionList[action]
            # sheepAction = np.asarray((1,1))

            # print (action, sheepAction)

            wolfAction = takeWolfAction(oldStates, wolfPrecision)
            distractorAction = takeDistractorAction(oldStates)

            currentActions = [sheepAction, wolfAction, distractorAction]
            # print (currentActions)
            
            currentStates = transState(oldStates, currentActions)
            currentBelief = updateBelief(oldBelief, currentStates)

            currentBelief_input = np.asarray(currentBelief).flatten()
            reward = beliefReward(currentBelief_input, currentActions)

            # print (reward)

            if isTerminals(currentBelief_input):
                done = 1
            else:
                done = 0

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

        # pygame viz
            import pygame
            from pygame.color import THECOLORS
            from pygame.locals import *

            agent_state = oldBelief_input.flatten()[:5]
            wolf_state = oldBelief_input.flatten()[5:10]
            distractor_state = oldBelief_input.flatten()[10:15]

            agent_coordinates = list(agent_state[:2])
            wolf_coordinates = list(wolf_state[:2])
            distractor_coordinates = list(distractor_state[:2])

            agent_coordinates = list(map(int, agent_coordinates))
            wolf_coordinates = list(map(int, wolf_coordinates))
            distractor_coordinates = list(map(int, distractor_coordinates))

            pygame.init()
            #screen_size = [np.multiply(movingRange[2],30),np.multiply(movingRange[3],30)]
            screen_size = [movingRange[2],movingRange[3]]
            screen=pygame.display.set_mode(screen_size)
            circleR = 10
            screen.fill([0,0,0])
            color = [THECOLORS['green'],THECOLORS['red'],THECOLORS['blue']]

            position_list = [agent_coordinates, wolf_coordinates, distractor_coordinates]

            # print(position_list)
            print(reward)

            for drawposition in position_list:
            	pygame.draw.circle(screen,color[int(position_list.index(drawposition))],drawposition,circleR)
                #pygame.draw.circle(screen,color[int(position_list.index(drawposition))],[np.int(np.multiply(i,30)) for i in drawposition],circleR)
            pygame.display.flip()
            #pygame.time.wait(0.2)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()


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
