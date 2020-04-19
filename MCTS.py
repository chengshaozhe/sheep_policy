import math
import numpy as np
import tensorflow as tf
import numpy as np

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 200
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
        self.Es = {}        # stores game ended for s
        self.action_size = action_size
        self.state_size = state_size

    def __call__(self, state):
        action = np.argmax(self.getActionProb(state))
        return action

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


    def search(self, state):
        s = state

        if s not in self.Es:
            self.Es[s] = isTerminal(s)

        if self.Es[s] != 0:
            return -500

    #Expand
        if s not in self.Ps:
            # leaf node
            self.Ps[s] = self.actor.getQ(np.reshape(np.asarray(s), [1, self.state_size]))
            # v = self.dqn.getV(np.reshape(np.asarray(s), [1, self.state_size]))
            s_arr = np.reshape(np.asarray(s), [1, self.state_size])
            v = self.critic(s_arr, actor)
            self.Ns[s] = 0
            return v

        current_best = -float('inf')
        best_action = -1

    #Selection
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

    #Evaluation/simulation
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

    #BackUp
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
    


    dqn = DQNAgent(state_size, action_size)
    dqn.load("./save/SingleWolf_episode_3000.h5")

    actor = ActorNetwork(state_size, action_size)
    actor.model.load_weights(
        './ddpg_save/SingleWolf_DDPG_episode_800-actormodel.h5')
    critic = CriticNetwork(state_size, action_size)
    critic.model.load_weights('./ddpg_save/SingleWolf_DDPG_episode_800-criticmodel.h5')
    
    cpuct = 1
    numMCTSSims = 10
    mcts = MCTS(actor, critic, cpuct, numMCTSSims, state_size, action_size)
    state = (80, 30, 0, 0, 60, 60, 0, 0)
    a = np.argmax(mcts.getActionProb(state))

    # print(a)
    # print(sheepActionList[a])

    num_opisodes = 1000
    animation = 1
    for e in range(num_opisodes):
        score = 0

        init_positionList = initialPosition(numberObjects)
        # print(init_positionList)
        if init_positionList == False:
            continue

        statesList = []
        initVelocity = [0, 0]
        for initPosition in init_positionList:
            statesList.append(initPosition + initVelocity)

        oldStates = pd.DataFrame(statesList, index=list(range(numberObjects)),
                                 columns=['positionX', 'positionY', 'velocityX', 'velocityY'])

        # wolfPrecision = random.choice(assumeWolfPrecisionList)# [50,11,3.3,1.83,0.92,0.31]

        total_reward = 0

        for wolfPrecision in assumeWolfPrecisionList:
            wolfPrecision = wolfPrecision
            done = False
            mcts = MCTS(actor, critic, cpuct, numMCTSSims,
                        state_size, action_size)
            for time in range(1000):
                loss = 0
                action_input = np.zeros([1, action_size])

                oldStates_array = np.asarray(oldStates).flatten()
                oldStates_input = np.reshape(oldStates_array, [1, state_size])

                # epsilon -= 1.0 / EXPLORE
                # epsilon = 0
                # if np.random.rand() <= epsilon:
                #     action_index = random.randrange(action_size)
                # else:
                #     action_index = actor(oldStates_input)

                # action_input[0][action_index] = 1
                stateList = list(oldStates_array)
                action_index = np.argmax(mcts.getActionProb(stateList))

                sheepAction = sheepActionList[action_index]
                wolfAction = takeWolfAction(oldStates, wolfPrecision)

                currentActions = [sheepAction, wolfAction]

                currentStates = transState(oldStates, currentActions)
                currentStates_array = np.asarray(currentStates).flatten()

                # reward = stateReward(currentStates_array,
                #                      currentActions, movingRange, time)

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
                    screen_size = [movingRange[2], movingRange[3]]
                    screen = pygame.display.set_mode(screen_size)
                    circleR = 10
                    screen.fill([0, 0, 0])
                    color = [THECOLORS['green'], THECOLORS['red']] + \
                        [THECOLORS['blue']] * (numberObjects - 2)
                    position_list = [agent_coordinates, wolf_coordinates]

                    # print(reward)

                    for drawposition in position_list:
                        pygame.draw.circle(
                            screen, color[int(position_list.index(drawposition))], drawposition, circleR)
                    pygame.display.flip()
                    # pygame.time.wait(0.2)

                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                if isTerminal(currentStates_array):
                    done = 1
                else:
                    done = 0

                currentStates_input = np.reshape(
                    currentStates_array, [1, state_size])

                # total_reward += reward
                oldStates = currentStates

                if done:
                    score = time
                    print (score)
                    break
                if time == 999:
                    print ("score:", 999)
