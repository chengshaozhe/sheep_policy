import tensorflow as tf
import numpy as np
import random
from collections import deque
import functools as ft
import os
import csv
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten,LSTM,BatchNormalization,Activation
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
    #     model.compile(loss='mse',
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

def isTerminals(state,time):
    agent_state = state[:4]
    wolf_state = state[4:8]

    agent_coordinates = agent_state[:2]
    wolf_coordinates = wolf_state[:2]

    if l2_norm(agent_coordinates, wolf_coordinates) <= 30 and time > 240:
        return True
    return False

def beliefReward(state, action, movingRange, time):
    if isTerminals(state,time):
        return -500

    agent_state = state[:4]

    target_states =  state[4:24]
    target_list = [target_states[i:i+4] for i in range(0,len(target_states),4)]
    distance_list = [l2_norm(agent_state[:2], target_state[:2]) for target_state in target_list]

    belief_states = state[24:]
    belief_states_list = [belief_states[i:i+6] for i in range(0,len(belief_states),6)]
    wolf_prob_list = [np.sum(probs) for probs in belief_states_list]

    def linear_punish(distance, prob, const = -500):
        punish = const / distance * prob
        return punish

    def linear_reward(distance, prob, const = 0.01):
        reward =  distance * np.exp(prob) * const
        return reward

    # wolf_punish = np.sum([linear_punish(distance, prob) for (distance, prob) in zip(distance_list, wolf_prob_list)])
   
    def signod_barrier(x,  m=1, s=1):
        expt_term = np.exp(-s*x)
        result =  m/(1.0+expt_term) 
        return result

    def barrier_punish_sigmod(pos, movingRange):
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
        x, y = position
        if x < 30 or x > movingRange[-1]-30:
            wall_punish -= min(500/x, 100)
        if y < 30 or y > movingRange[-1]-30:
            wall_punish -= min(500/y, 100)
        return wall_punish 


    def time_reward(time, const=1):
        if time == 999:
            reward = 500
        else:
            reward =  const
        return reward


    # wall_punish = barrier_punish_sigmod(state[:2], movingRange)
    wall_punish = wall_punish(state[:2], movingRange)
    survive_reward = time_reward(time)
    distance_reward = np.sum([linear_reward(distance, prob) for (distance, prob) in zip(distance_list, wolf_prob_list)])
    
    # print(wall_punish, distance_reward, survive_reward)
    return wall_punish + distance_reward + survive_reward


if __name__ == '__main__':
    statesListInit = [[10,10,0,0],[10,5,0,0],[15,15,0,0]]
    speedList = [8,4,4,4,4,4]
    movingRange = [0,0,364,364]
    assumeWolfPrecisionList = [50,11,3.3,1.83,0.92,0.31]
    sheepIdentity = 0
    wolfIdentity = 1

    distractorIdentity = 2
    distractor2_Identity = 3
    distractor3_Identity = 4
    distractor4_Identity = 5

    distractorPrecision = 0.5 / 3.14
    maxDistanceToFixation = movingRange[3]
    minDistanceEachOther = 50
    maxDistanceEachOther = 180
    minDistanceWolfSheep = 120

    numberObjects = 6

    PureAttentionModel = 1
    HybridModel = 0
    IdealObserveModel = 0 

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

    initialPosition=InitialPosition(movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
    transState = Transition.Transition(movingRange, speedList)
    computePrecisionAndDecay = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
    switchAttention = Attention.AttentionSwitch(attentionLimitation)
    updateBelief = BeliefUpdateWithAttention(computePrecisionAndDecay, switchAttention, attentionSwitchFrequency, sheepIdentity)

    takeWolfAction = WolfPolicy(sheepIdentity, wolfIdentity, speedList[wolfIdentity])
    takeDistractorAction = DistractorPolicy(distractorIdentity, distractorPrecision, speedList[distractorIdentity])
    takeDistractorAction2 = DistractorPolicy(distractor2_Identity, distractorPrecision, speedList[distractor2_Identity])
    takeDistractorAction3 = DistractorPolicy(distractor3_Identity, distractorPrecision, speedList[distractor3_Identity])
    takeDistractorAction4 = DistractorPolicy(distractor4_Identity, distractorPrecision, speedList[distractor4_Identity])


    numOfActions = 16
    actionAnglesList = [i * (360 / numOfActions)
                        for i in range(1, numOfActions + 1)]

    sheepActionList = [np.array((speedList[sheepIdentity] * np.cos(actionAngles * np.pi / 180),
                    speedList[sheepIdentity] * np.sin(actionAngles * np.pi / 180))) for actionAngles in actionAnglesList]


    

    state_size = 8
    action_size = numOfActions
    agent = DQNAgent(state_size, action_size)


    test_policy_name = 'SingleWolf_episode_3000.h5'
    agent.load("./save/" + test_policy_name) 
    
    loss_log = []
    score_log = []
    acc_log = []
    escape_rate_log = []
    batch_size = 64
    replay_start_size = 1000
    num_opisodes = 10

    done = False

    for wolfPrecision in assumeWolfPrecisionList:
        total_score = []
        total_acc = []
        episode_acc = []
        escape_count = 0
        escape_rate = 0
        for e in range(num_opisodes):
            score = 0
            step_acc = []
            init_positionList = initialPosition(numberObjects)
            if init_positionList == False:
                continue

            statesList = []
            initVelocity = [0,0]
            for initPosition in init_positionList:
                statesList.append(initPosition+initVelocity)

            oldStates = pd.DataFrame(statesList,index=list(range(numberObjects)),
                columns=['positionX','positionY','velocityX','velocityY'])

            oldBelief = initiateBeliefDF(numberObjects, assumeWolfPrecisionList)

            oldAttentionStatus = initiateAttentionStatus(oldBelief, attentionLimitation)

            wolfPrecision = wolfPrecision
            # print(wolfPrecision)

            done = False
            for time in range(1000):
                oldStates_array = np.asarray(oldStates).flatten()
                oldBelief_array = np.asarray(oldBelief).flatten()
                oldBelief_input = np.concatenate((oldStates_array, oldBelief_array))


                belief_states = oldBelief_input.flatten()[24:]
                belief_states_list = [belief_states[i:i+6] for i in range(0,len(belief_states),6)]
                wolf_prob_list = [np.sum(probs) for probs in belief_states_list]

                sheep_state = oldStates_array[:4]
                target_states = oldStates_array[4:24]
                target_states_list = [target_states[i:i+4] for i in range(0,len(target_states),4)]

                action_value_list = []

                for (i,target_state) in enumerate(target_states_list):
                    agent_input = np.concatenate((sheep_state, target_state))
                    agent_input = np.reshape(agent_input,[1,state_size])
                    wighted_Q = agent.getQ(agent_input) * wolf_prob_list[i]
                    action_value_list.append(wighted_Q) 

                action_value_array = np.asarray(action_value_list)
                weighted_action_value = np.sum(action_value_array, axis=0)
                action = np.argmax(weighted_action_value)

                # wolf_index = np.argmax(wolf_prob_list)

                # count = np.random.multinomial(1, wolf_prob_list)
                # wolf_index = np.argmax(count)
                # print (wolf_index)

                # if wolf_index==0:
                #     print('I SEE YOU!!!')
                # else:
                #     print("OPPS!!!")

                # agent_input = oldBelief_input[4*(wolf_index+1):4*(wolf_index+1)+4]
                # agent_input = np.concatenate((oldBelief_input[:4], agent_input))
                # agent_input= np.reshape(agent_input,[1,state_size])

                # action = agent(agent_input)

                sheepAction = sheepActionList[action]
                wolfAction = takeWolfAction(oldStates, wolfPrecision)
                distractorAction =  np.array(initVelocity)
                distractorAction2 =  np.array(initVelocity)
                distractorAction3 =  np.array(initVelocity)
                distractorAction4 =  np.array(initVelocity)

                distractor_update_step = 20

                if time % distractor_update_step == 0:
                    distractorAction = takeDistractorAction(oldStates)
                    distractorAction2 = takeDistractorAction2(oldStates)
                    distractorAction3 = takeDistractorAction3(oldStates)
                    distractorAction4 = takeDistractorAction4(oldStates)

                currentActions = [sheepAction, wolfAction, distractorAction, 
                                    distractorAction2, distractorAction3, distractorAction4]
                
                currentStates = transState(oldStates, currentActions)
                [currentBelief, currentAttentionStatus] = updateBelief(oldBelief, oldStates, currentStates, oldAttentionStatus, time+1)
                
                currentStates_array = np.asarray(currentStates).flatten()
                currentBelief_array = np.asarray(currentBelief).flatten()
                currentBelief_input = np.concatenate((currentStates_array, currentBelief_array))

                reward = beliefReward(currentBelief_input, currentActions, movingRange, time)

                if isTerminals(currentBelief_input, time):
                    done = 1
                else:
                    done = 0

                oldStates = currentStates
                oldBelief = currentBelief
                oldAttentionStatus = currentAttentionStatus


                beliefACC = np.sum(oldBelief_input.flatten()[24::5])

                step_acc.append(beliefACC)
                episode_acc = np.mean(step_acc)

                if done:
                    print("episode: {}/{}, score: {}, beliefACC: {}"
                          .format(e, num_opisodes, time, episode_acc))

                    score = time
                    total_score.append([score])
                    total_acc.append([episode_acc])
                    break
                
                if time == 999:
                    escape_count += 1
                    total_score.append([999])
                    total_acc.append([episode_acc])


                    print("episode: {}/{}, score: {}, beliefACC: {}"
                          .format(e, num_opisodes, time, episode_acc))

        escape_rate = escape_count / len(total_score)
        escape_rate_log.append([escape_rate])
        acc_log.append([np.mean(total_acc)])
        score_log.append([np.mean(total_score)])

        print('wolfPrecision:', wolfPrecision)
        print('acc:', acc_log)
        print('score:', score_log)
        print('escape:', escape_rate_log)
        print('=========================================')

    filename = test_policy_name + '-'  + str(e) + 'test' + "-PureAttentionModel"
    with open('analysis/acc-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for acc_item in acc_log:
            wr.writerow(acc_item)

    with open('analysis/score-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for score_item in score_log:
            wr.writerow(score_item)

    with open('analysis/escape_rate-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for escape_rate in escape_rate_log:
            wr.writerow(escape_rate)

    print("filename:", filename)
    # module_path = os.path.dirname(os.path.abspath(__file__))
    # filename = 'IdealObserveModel-episode-' + str(e)
    # logResults(filename, score_log)

