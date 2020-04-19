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
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, BatchNormalization,Activation,Input,merge,Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential, Model
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


from keras.layers import Dense, Flatten, Input, merge, Lambda


import math
import numpy as np
import tensorflow as tf
import numpy as np


EPS = 1e-8
import Transition
from PreparePolicy import *
import pandas as pd

from viz import *
from reward import *
from gridworld import *
from BeliefUpdate import *
from PreparePolicy import *
from InitialPosition import *
import Attention
import Transition

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 200


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0
        self.learning_rate = 0.001
        self.model = self._buildDNN()

    def _buildDNN(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(
            400, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(300, activation='relu'))
        model.add(tf.keras.layers.Dense(
            self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def getQ(self, state):
        action_values = self.model.predict(state)
        return action_values[0]

    def getV(self, state):
        action_values = self.model.predict(state)[0]
        state_value = np.amax(action_values)
        return state_value

    def load(self, name):
        self.model.load_weights(name)

    def __call__(self, state):
        action_values = self.model.predict(state)
        return action_values

class CriticNetwork:
    def __init__(self,state_size, action_size):
        self.model = self.create_critic_network(state_size, action_size)
        
    def create_critic_network(self, state_size, action_size):
        S = tf.keras.layers.Input(shape=[state_size])  
        A = tf.keras.layers.Input(shape=[action_size],name='action2')   
        w1 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = tf.keras.layers.Add()([h1,a1])
        h3 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = tf.keras.layers.Dense(action_size, activation='linear')(h3)   
        model = tf.keras.models.Model(inputs=[S,A],outputs=[V])
        return model

    def __call__(self, states, actor):
        action_values = self.model.predict([states, actor.model.predict(states)])
        state_value = np.amax(action_values)
        return state_value

class ActorNetwork:
    def __init__(self, state_size, action_size):
        self.model = self.create_actor_network(state_size, action_size)

    def create_actor_network(self, state_size, action_size):
        S = tf.keras.layers.Input(shape=[state_size])
        h0 = tf.keras.layers.Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = tf.keras.layers.Dense(HIDDEN2_UNITS, activation='relu')(h0)
        out = tf.keras.layers.Dense(action_size, activation='softmax')(h1)
        model = tf.keras.models.Model(inputs=[S], outputs=[out])
        return model

    def getQ(self, state):
        action_values = self.model.predict(state)[0]
        return list(action_values)

    def getV(self, state):
        action_values = self.model.predict(state)[0]
        state_value = np.amax(action_values)
        return state_value

    def __call__(self, state):
        action_values = self.model.predict(state)[0]
        action_index_max = np.argmax(action_values)
        return action_index_max



class MCTS():
    def __init__(self, actor, critic, cpuct, numMCTSSims, state_size, action_size):
        self.actor = actor
        self.critic = critic
        self.cpuct = cpuct
        self.numMCTSSims = numMCTSSims
        self.Qsa = {}       # stores Q values for s,a
        self.Nsa = {}       # stores #times edge s,a was visited
        self.Ns = {}        # stores #times board s was visited
        self.Ps = {}        # stores initial policy (returned by neural net) {s:{a:prob}}
        self.Es = {}        # stores game ended for  s
        self.action_size = action_size
        self.state_size = state_size

    def getActionProb(self, state):
        for i in range(self.numMCTSSims):
            state = tuple(map(int, state))
            self.search(state)

        s = state
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.action_size)]

        # print(counts)
        bestA = np.argmax(counts)
        probs = [0] * len(counts)
        probs[bestA] = 1

        return probs

    def getActionCounts(self, state):

        for i in range(self.numMCTSSims):
            state = tuple(map(int, state))
            self.search(state)

        s = state
        counts = [self.Nsa[(s, a)] if (
            s, a) in self.Nsa else 0 for a in range(self.action_size)]
        return counts

    def search(self, state):
        s = state

        if s not in self.Es:
            self.Es[s] = isTerminal(s)

        if self.Es[s] != 0:
            return -500

        if s not in self.Ps:
            self.Ps[s] = self.actor.getQ(np.reshape(np.asarray(s), [1, self.state_size]))
            # v = self.dqn.getV(np.reshape(np.asarray(s), [1, self.state_size]))
            s_arr = np.reshape(np.asarray(s), [1, self.state_size])
            v = self.critic(s_arr, actor)

            self.Ns[s] = 0
            return v

        current_best = -float('inf')
        best_action = -1

        for a in range(action_size):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * \
                    math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

            if u > current_best:
                current_best = u
                best_action = a

        a = best_action

        sheepAction = sheepActionList[a]
        statesList = [list(s[:4]), list(s[4:])]
        state_pd = pd.DataFrame(statesList, index=list(range(2)),
                                columns=['positionX', 'positionY', 'velocityX', 'velocityY'])

        wolfAction = takeWolfAction(state_pd, 50)
        currentActions = [sheepAction, wolfAction]
        next_state = transState(state_pd, currentActions)
        next_state = tuple(np.asarray(next_state).flatten())
        next_state = tuple(map(int, next_state))
        v = self.search(next_state)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1

        return v


def isTerminal(state):
    agent_state = state[:4]
    wolf_state = state[4:8]

    agent_coordinates = agent_state[:2]
    wolf_coordinates = wolf_state[:2]

    def l2_norm(s0, s1, rho=1):
        diff = (np.asarray(s0) - np.asarray(s1)) * rho
        return np.linalg.norm(diff)

    if l2_norm(agent_coordinates, wolf_coordinates) <= 30:
        return True

    return False

def beliefReward(state, action, movingRange, time):
    if isTerminals(state, time):
        return -500

    agent_state = state[:4]

    target_states =  state[4:24]
    target_list = [target_states[i:i+4] for i in range(0,len(target_states),4)]
    distance_list = [l2_norm(agent_state[:2], target_state[:2]) for target_state in target_list]

    belief_states = state[24:]
    belief_states_list = [belief_states[i::5] for i in range(0,len(belief_states),6)]
    wolf_prob_list = [np.sum(probs) for probs in belief_states_list]

    def linear_punish(distance, prob, const = -500):
        punish = const / distance * prob
        return punish

    wolf_punish = np.sum([linear_punish(distance, prob) for (distance, prob) in zip(distance_list, wolf_prob_list)])
   
    def signod_barrier(x,  m=1, s=1):
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
        reward = min(np.min(punishment, axis=-1) - 11, 1)
        return reward

    wall_punish  = barrier_punish(state[:2], movingRange)

    # print(wolf_punish, wall_punish)
    return wolf_punish + wall_punish

def stateReward(state, action, movingRange,time):
    if isTerminals(state):
        return -500
    agent_state = state[:4]
    wolf_state = state[4:8]
    distance = l2_norm(agent_state[:2], wolf_state[:2])

    def signod_barrier(x,  m=1, s=1):
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
            reward =  0
        return reward


    # wall_punish = barrier_punish(state[:2], movingRange)
    wall_punish = wall_punish(state[:2], movingRange)
    survive_reward = time_reward(time)
    distance_reward = distance * 0.01

    # print(wall_punish, distance_reward, survive_reward)
    return wall_punish + distance_reward + survive_reward

def isTerminals(state, time):
    agent_state = state[:4]
    wolf_state = state[4:8]

    agent_coordinates = agent_state[:2]
    wolf_coordinates = wolf_state[:2]

    if l2_norm(agent_coordinates, wolf_coordinates) <= 30 and time > 240:
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

    PureAttentionModel = 0
    HybridModel = 1
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

    initialPosition = InitialPosition(movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
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

    epsilon = 1

    actor = ActorNetwork(state_size, action_size)
    actor.model.load_weights(
        './ddpg_save/SingleWolf_DDPG_episode_800-actormodel.h5')
    critic = CriticNetwork(state_size, action_size)
    critic.model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_800-criticmodel.h5')
    

    loss_log = []
    acc_log = []
    score_log = []
    escape_rate_log = []

    num_opisodes = 10
    cpuct = 1
    numMCTSSims = 10
    animation = 1

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
            # wolfPrecision = 0.5 / 3.14
            # print(wolfPrecision)

            done = False
            for time in range(1000):
                oldStates_array = np.asarray(oldStates).flatten()
                oldBelief_array = np.asarray(oldBelief).flatten()
                oldBelief_input = np.concatenate((oldStates_array, oldBelief_array))


                belief_states = oldBelief_input.flatten()[24:]
                belief_states_list = [belief_states[i::5] for i in range(0,len(belief_states),6)]
                wolf_prob_list = [np.sum(probs) for probs in belief_states_list]


                sheep_state = oldStates_array[:4]
                target_states = oldStates_array[4:24]
                target_states_list = [target_states[i:i+4] for i in range(0,len(target_states),4)]

                action_counts_array = np.zeros((1,action_size))

                for (i,target_state) in enumerate(target_states_list):
                    agent_input = np.concatenate((sheep_state, target_state))
                    mcts = MCTS(actor, critic, cpuct, numMCTSSims, state_size, action_size)
                    stateList = list(agent_input)
                    action_counts = mcts.getActionCounts(stateList) 
                    weightd_counts = [x * wolf_prob_list[i] for x in action_counts]
                    action_counts_array +=  np.array(weightd_counts)
                    # print(action_counts_array)
                # print (action_counts_array)
                action_index = np.argmax(action_counts_array)

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

                # action = actor(agent_input)
                # action_index = random.randrange(action_size)

                # mcts = MCTS(actor, critic, cpuct, numMCTSSims, state_size, action_size)
                # stateList = list(agent_input)
                # action_index = np.argmax(mcts.getActionProb(stateList))

                sheepAction = sheepActionList[action_index]
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

                if animation:
                    import pygame
                    from pygame.color import THECOLORS
                    from pygame.locals import *
                    agent_state = oldBelief_input.flatten()[:4]
                    wolf_state = oldBelief_input.flatten()[4:8]
                    distractor_state = oldBelief_input.flatten()[8:12]
                    distractor_state2 = oldBelief_input.flatten()[12:16]
                    distractor_state3 = oldBelief_input.flatten()[16:20]
                    distractor_state4 = oldBelief_input.flatten()[20:24]

                    agent_coordinates = list(agent_state[:2])
                    wolf_coordinates = list(wolf_state[:2])
                    distractor_coordinates = list(distractor_state[:2])
                    distractor_coordinates2 = list(distractor_state2[:2])
                    distractor_coordinates3 = list(distractor_state3[:2])
                    distractor_coordinates4 = list(distractor_state4[:2])

                    agent_coordinates = list(map(int, agent_coordinates))
                    wolf_coordinates = list(map(int, wolf_coordinates))
                    distractor_coordinates = list(map(int, distractor_coordinates))
                    distractor_coordinates2 = list(map(int, distractor_coordinates2))
                    distractor_coordinates3 = list(map(int, distractor_coordinates3))
                    distractor_coordinates4 = list(map(int, distractor_coordinates4))

                    pygame.init()
                    screen_size = [movingRange[2],movingRange[3]]
                    screen = pygame.display.set_mode(screen_size)
                    circleR = 10
                    screen.fill([0,0,0])
                    color = [THECOLORS['green'],THECOLORS['red']] + [THECOLORS['blue']] * (numberObjects-2)
                    position_list = [agent_coordinates, wolf_coordinates, distractor_coordinates,
                                    distractor_coordinates2, distractor_coordinates3, distractor_coordinates4]

                    # print(position_list)
                    # print(reward)

                    for drawposition in position_list:
                        pygame.draw.circle(screen,color[int(position_list.index(drawposition))],drawposition,circleR)
                    pygame.display.flip()
                    #pygame.time.wait(0.2)

                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()


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

    filename = 'SingleWolf_DDPG_episode_' + str(Test_wight_name) + '-'  + str(e) + 'test' + "-HybridModel"
    with open('analysis_ddpg/acc-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for acc_item in acc_log:
            wr.writerow(acc_item)

    with open('analysis_ddpg/score-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for score_item in score_log:
            wr.writerow(score_item)

    with open('analysis_ddpg/escape_rate-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for escape_rate in escape_rate_log:
            wr.writerow(escape_rate)

    print("filename:", filename)




