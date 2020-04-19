import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools as ft
import os
import csv
from PIL import Image
import json

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, BatchNormalization,Activation
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

import argparse


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=40000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._buildDNN()
        self.target_model = self._buildDNN()
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

    # def _buildDNN(self):
    #     model = Sequential()
    #     model.add(Dense(400, input_dim=self.state_size))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #     model.add(Dense(300))
    #     model.add(BatchNormalization())
    #     model.add(Activation('relu'))
    #     model.add(Dense(self.action_size, activation='softmax'))
    #     model.compile(loss=self._huberLoss,
    #                   optimizer=Adam(lr=self.learning_rate))
    #     return model

    def _buildDNN(self):
        model = Sequential()
        model.add(Dense(400, input_dim=self.state_size, activation='relu'))
        model.add(Dense(300, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss = 'mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def _buildRNN(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(1, self.state_size), return_sequences=False))
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
        # tensorboard = TensorBoard(log_dir='./logs')
        # history = self.model.fit(states_mb, targets_mb, epochs=1, verbose=0,callbacks=[tensorboard])
        history = self.model.fit(states_mb, targets_mb, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return history.history['loss']

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def __call__(self, state):
        action_values = self.model.predict(state)
        action_index_max = np.argmax(action_values[0])
        return action_index_max


def logResults(filename, loss_log, score_log):
    with open('results/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

    with open('results/reward_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for score_item in score_log:
            wr.writerow(score_item)


def stateReward(state, action, movingRange,time):
    if isTerminals(state):
        return -500

    agent_state = state[:4]
    wolf_state = state[4:8]
    distance = l2_norm(agent_state[:2], wolf_state[:2])

    def signod_barrier(x, m=1, s=1):
        expt_term = np.exp(-s*x)
        result =  m/(1.0+expt_term) 
        return result

    def barrier_punish(pos, movingRange):
        sn_arr = np.asarray(pos)

        lower = np.asarray(movingRange[:2])
        upper = np.asarray(movingRange[2:])
        margin = np.array([0, 0])
        to_upper = upper - sn_arr + margin
        to_lower = sn_arr - lower - margin

        punishment = signod_barrier(np.hstack([to_upper, to_lower]), m = 15, s = 1/30)
        reward = min(np.sum(punishment, axis=-1) - 54, 0.1)
        return reward

    def wall_punish(position, movingRange):
        wall_punish = 0
        x,y = position
        if x < 30 or x > movingRange[-1]-30:
            wall_punish -= min(1000/x, 100)
        if y < 30 or y > movingRange[-1]-30:
            wall_punish -= min(1000/y, 100)
        return wall_punish 

    def time_reward(time, const=0.1):
        if time == 999:
            reward = 500
        else:
            reward =  const
        return reward

    # wall_punish = barrier_punish(state[:2], movingRange)
    wall_punish = wall_punish(state[:2], movingRange)
    survive_reward = time_reward(time)
    distance_reward = distance * 0.1

    print(wall_punish, distance_reward, survive_reward)
    return wall_punish + distance_reward + survive_reward

def isTerminals(state):
    agent_state = state[:4]
    wolf_state = state[4:8]

    agent_coordinates = agent_state[:2]
    wolf_coordinates = wolf_state[:2]

    if l2_norm(agent_coordinates, wolf_coordinates) <= 30:
        return True
    return False

if __name__ == '__main__':
    statesListInit = [[10,10,0,0],[10,5,0,0],[15,15,0,0]]
    speedList = [8,4,4,4,4,4]
    movingRange = [0,0,364,364]
    assumeWolfPrecisionList = [50,11,3.3,1.83,0.92,0.31]
    circleR = 10
    sheepIdentity = 0
    wolfIdentity = 1

    distractorPrecision = 0.5 / 3.14
    maxDistanceToFixation = movingRange[3]
    minDistanceEachOther = 50
    maxDistanceEachOther = 180
    minDistanceWolfSheep = 120

    numberObjects = 2

    PureAttentionModel = 0
    HybridModel = 0
    IdealObserveModel = 1 


    if PureAttentionModel:
        attentionLimitation = 2
        precisionPerSlot = 8.0
        precisionForUntracked = 0
        memoryratePerSlot = 0.7
        memoryrateForUntracked = 0

    if HybridModel:
        attentionLimitation = 2
        precisionPerSlot = 8
        precisionForUntracked = 2.5
        memoryratePerSlot = 0.7
        memoryrateForUntracked = 0.45

    if IdealObserveModel:
        attentionLimitation = 100
        precisionPerSlot = 50
        precisionForUntracked = 50
        memoryratePerSlot = 0.99
        memoryrateForUntracked = 0.99

    attentionSwitchFrequency = 12

    initialPosition = InitialPosition(movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
    transState = Transition.Transition(movingRange, speedList)
    computePrecisionAndDecay = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
    switchAttention = Attention.AttentionSwitch(attentionLimitation)
    updateBelief = BeliefUpdateWithAttention(computePrecisionAndDecay, switchAttention, attentionSwitchFrequency, sheepIdentity)

    takeWolfAction = WolfPolicy(sheepIdentity, wolfIdentity, speedList[wolfIdentity])
   

    numOfActions = 16
    actionAnglesList = [i * (360 / numOfActions)
                        for i in range(1, numOfActions + 1)]

    sheepActionList = [np.array((speedList[sheepIdentity] * np.cos(actionAngles * np.pi / 180),
                    speedList[sheepIdentity] * np.sin(actionAngles * np.pi / 180))) for actionAngles in actionAnglesList]


    state_size = numberObjects * 4 
    action_size = numOfActions
    agent = DQNAgent(state_size, action_size)

    # agent.load("./save/SingleWolf_episode_3000.h5") 

    loss_log = []
    score_log = []

    batch_size = 64
    replay_start_size = 1000
    num_opisodes = 100001

    for e in range(num_opisodes):
        score = 0

        init_positionList = initialPosition(numberObjects)
        # print(init_positionList)
        if init_positionList == False:
            continue

        statesList = []
        initVelocity = [0,0]
        for initPosition in init_positionList:
            statesList.append(initPosition + initVelocity)

        # print (statesList)

        oldStates = pd.DataFrame(statesList,index=list(range(numberObjects)),
            columns=['positionX','positionY','velocityX','velocityY'])

        wolfPrecision = random.choice(assumeWolfPrecisionList)
        # wolfPrecision = 50
        done = False

        for time in range(1000):
            oldStates_array = np.asarray(oldStates).flatten()
            # oldStates_input = np.reshape(oldStates_array,[1,1,state_size]) # LSTM 
            oldStates_input = np.reshape(oldStates_array,[1,state_size])

            # action = agent.act(oldStates_input)
            action = agent.act(oldStates_input)

            sheepAction = sheepActionList[action]

            wolfAction = takeWolfAction(oldStates, wolfPrecision)
            currentActions = [sheepAction, wolfAction]
            
            currentStates = transState(oldStates, currentActions)
            currentStates_array = np.asarray(currentStates).flatten()

            reward = stateReward(currentStates_array, currentActions, movingRange,time)

            # print (reward)

        # pygame viz
            animation = 1
            if animation:
                import pygame
                from pygame.color import THECOLORS
                from pygame.locals import *
                agent_state = oldStates_input.flatten()[:4]
                wolf_state = oldStates_input.flatten()[4:8]
           
                agent_coordinates = list(agent_state[:2])
                wolf_coordinates = list(wolf_state[:2])
                
                agent_coordinates = list(map(int, agent_coordinates))
                wolf_coordinates = list(map(int, wolf_coordinates))
                
                pygame.init()
                screen_size = [movingRange[2],movingRange[3]]
                screen = pygame.display.set_mode(screen_size)
                circleR = 10
                screen.fill([0,0,0])
                color = [THECOLORS['green'],THECOLORS['red']] + [THECOLORS['blue']] * (numberObjects-2)
                position_list = [agent_coordinates, wolf_coordinates]

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

            if isTerminals(currentStates_array):
                done = 1
            else:
                done = 0

            # currentStates_input = np.reshape(currentStates_array,[1,1,state_size]) # LSTM input
            currentStates_input = np.reshape(currentStates_array,[1,state_size])

            agent.remember(oldStates_input, action, reward, currentStates_input, done)

            oldStates = currentStates

            if len(agent.memory) > replay_start_size:
                states_mb, targets_mb = agent.replay(batch_size)
                loss = agent.train(states_mb, targets_mb)

                if time % 10 == 0:

                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                          .format(e, num_opisodes, time, loss[0]))

                if time % 100 == 0:
                    loss_log.append(loss)


            if done:
                # agent.updateTargetModel()
                score = time
                break

        if e % 5 == 0:
            agent.updateTargetModel()


        score_log.append([score])

        if e % 10 == 0:
            module_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(module_path, "save")
            name = 'SingleWolf_episode_' + str(e) + '.h5'
            weight_path = os.path.join(data_path, name)
            agent.save(weight_path)

            filename = 'episode-' + str(e)
            logResults(filename, loss_log, score_log)
            loss_log = []
            score_log = []

        TEST = 0
        if TEST:
            if e % 100 == 0:
                test_episodes = 10
                score_log = []
                escape_rate_log = []
                done = False
                for wolfPrecision in assumeWolfPrecisionList:
                    total_score = []
                    escape_count = 0
                    escape_rate = 0
                    for test_episode in range(test_episodes):
                        score = 0
                        init_positionList = initialPosition(numberObjects)
                        # print(init_positionList)
                        if init_positionList == False:
                            continue

                        statesList = []
                        initVelocity = [0,0]
                        for initPosition in init_positionList:
                            statesList.append(initPosition+initVelocity)

                        oldStates = pd.DataFrame(statesList,index=list(range(numberObjects)),
                            columns=['positionX','positionY','velocityX','velocityY'])

                        wolfPrecision = wolfPrecision
                        done = False
                        for time in range(1000):
                            oldStates_array = np.asarray(oldStates).flatten()
                            # oldStates_input = np.reshape(oldStates_array,[1,1,state_size]) # LSTM 
                            oldStates_input = np.reshape(oldStates_array,[1,state_size])

                            action = agent(oldStates_input)
                            sheepAction = sheepActionList[action]

                            wolfAction = takeWolfAction(oldStates, wolfPrecision)
                            currentActions = [sheepAction, wolfAction]
                            
                            currentStates = transState(oldStates, currentActions)
                            currentStates_array = np.asarray(currentStates).flatten()

                            reward = stateReward(currentStates_array, currentActions, movingRange,time)


                            # print (reward)

                            if isTerminals(currentStates_array):
                                done = 1
                            else:
                                done = 0

                            # currentStates_input = np.reshape(currentStates_array,[1,1,state_size]) # LSTM input
                            currentStates_input = np.reshape(currentStates_array,[1,state_size])

                            oldStates = currentStates


                            if done:
                                print("episode: {}/{}, score: {}"
                                      .format(test_episode, test_episodes, time))

                                score = time
                                total_score.append([score])
                                break
                            
                            if time == 999:
                                escape_count += 1
                                total_score.append([999])


                                print("episode: {}/{}, score: {}"
                                      .format(test_episode, test_episodes, time))

                
                    escape_rate = escape_count / test_episodes 
                    escape_rate_log.append([escape_rate])
                    score_log.append([np.mean(total_score)])

                    print('wolfPrecision:', wolfPrecision)
                    print('score:', score_log)
                    print('escape:', escape_rate_log)
                    print('=========================================')

                filename = 'test_episode' + '-'  + str(e) + 'test' + str(test_episodes)
                with open('analysis/score-' + filename + '.csv', 'w') as lf:
                    wr = csv.writer(lf)
                    for score_item in score_log:
                        wr.writerow(score_item)

                with open('analysis/escape_rate-' + filename + '.csv', 'w') as lf:
                    wr = csv.writer(lf)
                    for escape_rate in escape_rate_log:
                        wr.writerow(escape_rate)

                print("filename:", filename)






