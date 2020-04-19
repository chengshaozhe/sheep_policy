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


def beliefReward(state, action, movingRange):
    if isTerminals(state):
        return -100

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

    # try:
    #     actor.model.load_weights("actormodel.h5")
    #     critic.model.load_weights("criticmodel.h5")
    #     actor.target_model.load_weights("actormodel.h5")
    #     critic.target_model.load_weights("criticmodel.h5")
    #     print("Weight load successfully")
    # except:
    #     print("Cannot find the weight")

    Test_wight_name = 800
    actor.model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_' + str(Test_wight_name) + "-actormodel.h5")
    critic.model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_' + str(Test_wight_name) + "-criticmodel.h5")
    actor.target_model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_' + str(Test_wight_name) + "-actormodel.h5")
    critic.target_model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_' + str(Test_wight_name) + "-criticmodel.h5")
  
    animation = 1
    TEST = 0
    
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

        oldStates = pd.DataFrame(statesList,index=list(range(numberObjects)),
            columns=['positionX','positionY','velocityX','velocityY'])

        wolfPrecision = random.choice(assumeWolfPrecisionList)
        # wolfPrecision = 50
        done = False
        total_reward = 0

        for time in range(1000):
            loss = 0 
            action_input = np.zeros([1,action_size])

            oldStates_array = np.asarray(oldStates).flatten()
            oldStates_input = np.reshape(oldStates_array,[1,state_size])

 
            action_index = actor(oldStates_input)

            action_input[0][action_index] = 1

            sheepAction = sheepActionList[action_index]
            wolfAction = takeWolfAction(oldStates, wolfPrecision)

            currentActions = [sheepAction, wolfAction]

            currentStates = transState(oldStates, currentActions)
            currentStates_array = np.asarray(currentStates).flatten()

            reward = stateReward(currentStates_array, currentActions, movingRange,time)

        # pygame viz
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

                # print(reward)

                for drawposition in position_list:
                    pygame.draw.circle(screen,color[int(position_list.index(drawposition))],drawposition,circleR)
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

            currentStates_input = np.reshape(currentStates_array,[1,state_size])

            memory.add(oldStates_array, action_input[0], reward, currentStates_array, done)


            batch = memory.getBatch(BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            dones = np.asarray([e[4] for e in batch])
            y_t = np.asarray([e[1] for e in batch])

            if len(memory.buffer) > replay_start_size:

                target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
                # target_q_values = critic(new_states, actor)

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + GAMMA*target_q_values[k]
           
                loss += critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()

 

                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, reward: {}, loss: {:.4f}"
                                  .format(e, num_opisodes, time, reward, loss))

                if time % 100 == 0:
                    loss_log.append([loss])

            total_reward += reward
            oldStates = currentStates

            if done:
                score = time
                print (score)
                break


        score_log.append([score])

        if np.mod(e, 10) == 0:
            print("Now we save model")
            module_path = os.path.dirname(os.path.abspath(__file__))
            data_path = os.path.join(module_path, "ddpg_save")
            name = 'SingleWolf_DDPG_episode_' + str(e) + '-'
            weight_path = os.path.join(data_path, name)

            actor.model.save_weights(weight_path + "actormodel.h5", overwrite=True)
            with open("ddpg_save/actormodel.json", "w") as outfile:
                json.dump(actor.model.to_json(), outfile)

            critic.model.save_weights(weight_path + "criticmodel.h5", overwrite=True)
            with open("ddpg_save/criticmodel.json", "w") as outfile:
                json.dump(critic.model.to_json(), outfile)

            filename = 'episode-' + str(e)
            logResults(filename, loss_log, score_log)
            loss_log = []
            score_log = []


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

                            action = actor(oldStates_input)
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

                
                    escape_rate = escape_count / len(total_score) 
                    escape_rate_log.append([escape_rate])
                    score_log.append([np.mean(total_score)])

                    print('wolfPrecision:', wolfPrecision)
                    print('score:', score_log)
                    print('escape:', escape_rate_log)
                    print('=========================================')

                filename = 'DDPG_test_episode' + '-'  + str(e) + 'test' + str(test_episodes)
                with open('analysis/score-' + filename + '.csv', 'w') as lf:
                    wr = csv.writer(lf)
                    for score_item in score_log:
                        wr.writerow(score_item)

                with open('analysis/escape_rate-' + filename + '.csv', 'w') as lf:
                    wr = csv.writer(lf)
                    for escape_rate in escape_rate_log:
                        wr.writerow(escape_rate)

                print("filename:", filename)






