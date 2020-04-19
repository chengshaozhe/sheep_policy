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


HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 200
BATCH_SIZE = 64
TAU = 0.9
LEARNING_RATE = 0.01

class CriticNetwork:
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_size):
        S = Input(shape=[state_size])  
        A = Input(shape=[action_size],name='action2')   
        w1 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A) 
        h1 = Dense(HIDDEN2_UNITS, activation='linear')(w1)
        h2 = Add()([h1,a1])
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(action_size, activation='linear')(h3)   
        model = Model(inputs=[S,A],outputs=[V])
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]


    def __call__(self, states, actor):
        target_value = self.target_model.predict([states, actor.target_model.predict(states)])
        return target_value

class ActorNetwork:
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        self.model, self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size, action_size):
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        out = Dense(action_size, activation='softmax')(h1)
        model = Model(inputs=[S], outputs=[out])
        return model, model.trainable_weights, S

    def chooseAction(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        action_values = self.model.predict(state)
        action_index_max = np.argmax(action_values[0])
        return action_index_max

    def __call__(self, state):
        action_values = self.model.predict(state)
        action_index_max = np.argmax(action_values[0])
        return action_index_max


class ReplayBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

def logResults(filename, loss_log, score_log):
    with open('results/loss_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)

    with open('results/reward_data-' + filename + '.csv', 'w') as lf:
        wr = csv.writer(lf)
        for score_item in score_log:
            wr.writerow(score_item)


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
    
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    TAU = 0.001
    LEARNING_RATE_ACTOR = 0.0001
    LEARNING_RATE_CRITIC = 0.001
    EXPLORE = 100000
    GAMMA = 0.99

    epsilon = 1
    batch_size = 64
    replay_start_size = 1000
    num_opisodes = 100001

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    actor = ActorNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE_ACTOR)
    critic = CriticNetwork(sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE_CRITIC)
    memory = ReplayBuffer(BUFFER_SIZE)

    loss_log = []
    score_log = []

    Test_wight_name = 800
    actor.model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_' + str(Test_wight_name) + "-actormodel.h5")
    critic.model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_' + str(Test_wight_name) + "-criticmodel.h5")
    actor.target_model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_' + str(Test_wight_name) + "-actormodel.h5")
    critic.target_model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_' + str(Test_wight_name) + "-criticmodel.h5")
  
    
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

                action_value_list = []

                for (i,target_state) in enumerate(target_states_list):
                    agent_input = np.concatenate((sheep_state, target_state))
                    agent_input = np.reshape(agent_input,[1,state_size])
                    wighted_Q = actor.model.predict(agent_input) * wolf_prob_list[i]
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

                # action = actor(agent_input)
                # action = random.randrange(action_size)

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




