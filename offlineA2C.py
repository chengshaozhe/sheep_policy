import tensorflow as tf
import numpy as np
import functools as ft
#import env
import dataSave 
import tensorflow_probability as tfp
import random
import pandas as pd

import BeliefUpdate
import PreparePolicy
import InitialPosition
import Attention

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

def BeliefArrayToDataFrame(objectsNumber, oldBelief, assumeWolfPrecisionList):
    multiIndex = pd.MultiIndex.from_product([assumeWolfPrecisionList, range(
        1, objectsNumber)], names=['assumeChasingPrecision', 'Identity'])
    oldBeliefList = list(oldBelief)
    beliefDF = pd.DataFrame(
        oldBeliefList, index=multiIndex, columns=['p'])
    return beliefDF

def PositionArraytToDataFrame(numberObjects, oldPosition):
    oldPositionList = oldPosition.tolist()
    oldPositionList = [oldPositionList[i:i+4] for i in range(0,len(oldPositionList),4)]
    oldPositionDF = pd.DataFrame(oldPositionList,index=list(range(numberObjects)),
    columns=['positionX','positionY','velocityX','velocityY'])
    return oldPositionDF

def renormalVector(rawVector, targetLength):
    rawLength = np.power(np.power(rawVector, 2).sum(), 0.5)
    changeRate = np.divide(targetLength, rawLength)
    return np.multiply(rawVector, changeRate)


class Reset():
    def __init__(self, numberObjects, initialPosition, assumeWolfPrecisionList, attentionLimitation):
        self.numberObjects = numberObjects
        self.initialPosition = initialPosition
        self.assumeWolfPrecisionList = assumeWolfPrecisionList
        self.attentionLimitation = attentionLimitation

    def __call__(self):
        initPositionList = self.initialPosition(self.numberObjects)
        # if initPositionList == False:
        #   continue
        statesList = []
        initVelocity = [0,0]
        for initPosition in initPositionList:
            statesList.append(initPosition + initVelocity)

        initState = pd.DataFrame(statesList,index=list(range(self.numberObjects)),
            columns=['positionX','positionY','velocityX','velocityY'])

        initBelief = BeliefUpdate.initiateBeliefDF(self.numberObjects, self.assumeWolfPrecisionList)
        initAttentionStatus = BeliefUpdate.initiateAttentionStatus(initBelief, self.attentionLimitation)

        initPhysicalkState = np.asarray(initState).flatten()
        initBelief = np.asarray(initBelief).flatten()

        initState = np.concatenate((initPhysicalkState, initBelief))
        return initState, initAttentionStatus

class RewardFunction():
    def __init__(self, aliveBouns):
        self.aliveBouns = aliveBouns

    def __call__(self, oldState, action):
        if isTerminal(oldState):
            return -20
        reward = self.aliveBouns
        return reward

def l2Norm(s0, s1, rho=1):
    diff = (np.asarray(s0) - np.asarray(s1)) * rho
    return np.linalg.norm(diff)

def isTerminal(state):
    agentState = state[:4]
    wolfState = state[4:8]

    agentCoordinates = agentState[:2]
    wolfCoordinates = wolfState[:2]

    if l2Norm(agentCoordinates, wolfCoordinates) <= 30:
        return True
    return False


def approximatePolicy(stateBatch, model):
    graph = model.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    actionSample_ = graph.get_tensor_by_name('outputs/actionSample_:0')
    actionBatch = model.run(actionSample_, feed_dict={state_: stateBatch})
    return actionBatch


class SampleTrajectory():
    def __init__(self, maxTimeStep, numberObjects, transitionFunction, isTerminal, reset, distractorUpdateStep, wolfPolicy, distractorPolicy, wolfPrecision,):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
        self.wolfPolicy = wolfPolicy
        self.wolfPrecision = wolfPrecision
        self.distractorPolicy = distractorPolicy
        self.distractorUpdateStep = distractorUpdateStep
        self.numberObjects = numberObjects

    def __call__(self, policy):
        oldState, oldAttentionStatus = self.reset()
        distractorAction =  np.array([0,0])
        trajectory = []
        scoreList = []
        beliefAccList = []
        for time in range(self.maxTimeStep):

            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = policy(oldStateBatch)
            action = actionBatch[0]

            # actionBatch shape: batch * action Dimension; only need action Dimention

            beliefIndex = 4 * self.numberObjects

            oldPosition = oldState[:beliefIndex,]
            oldPosition = PositionArraytToDataFrame(self.numberObjects, oldPosition)
            
            wolfAction = self.wolfPolicy(oldPosition, self.wolfPrecision)

            if time % self.distractorUpdateStep == 0:
                distractorAction = self.distractorPolicy(oldPosition)

            actionForTransition = [action, wolfAction, distractorAction]

            newState, newAttentionStatus = self.transitionFunction(
                oldState, actionForTransition, oldAttentionStatus, time)

            trajectory.append((oldState, action))

            terminal = self.isTerminal(newState)

            beliefACC = np.sum(oldState.flatten()[beliefIndex::5])
            # print('beliefACC:',beliefACC)

            if terminal:
                score = time
                beliefAccList.append([beliefACC])
                # print("score:", score)
                break

            if time == self.maxTimeStep - 1:
                score = self.maxTimeStep
                beliefAccList.append([beliefACC])
                # print("score:", score)

            oldState = newState
            oldAttentionStatus = newAttentionStatus

        return trajectory


class TransitionWithBelief():
    def __init__(self, movingRange, speedList,numberObjects, assumeWolfPrecisionList,updateBelief, renderOn=False):
        self.movingRange = movingRange
        self.speedList = speedList
        self.renderOn = renderOn
        self.numberObjects = numberObjects
        self.assumeWolfPrecisionList = assumeWolfPrecisionList
        self.updateBelief = updateBelief

    def __call__(self, oldState, actionForTransition, oldAttentionStatus, time):
        beliefIndex = 4 * self.numberObjects

        oldPosition = oldState[:beliefIndex,]
        oldPosition = PositionArraytToDataFrame(self.numberObjects, oldPosition)
        
        oldBelief = oldState[beliefIndex:,]
        oldBelief = BeliefArrayToDataFrame(self.numberObjects, oldBelief, self.assumeWolfPrecisionList)

        newPosition = self.physicalTransition(oldPosition, actionForTransition)

        [newBelief, newAttentionStatus] = self.updateBelief(
            oldBelief, oldPosition, newPosition, oldAttentionStatus, time + 1)

        newPositionArray = np.asarray(newPosition).flatten()
        newBeliefArray = np.asarray(newBelief).flatten()
        newState = np.concatenate((newPositionArray, newBeliefArray))

        return newState, newAttentionStatus

    def physicalTransition(self, currentStates, currentActions):

        currentPositions = currentStates.loc[:][[
            'positionX', 'positionY']].values
        currentVelocities = currentStates.loc[:][[
            'velocityX', 'velocityY']].values
        numberObjects = len(currentStates.index)


        newVelocities = [renormalVector(np.add(currentVelocities[i], np.divide(
            currentActions[i], 2.0)), self.speedList[i]) for i in range(numberObjects)]

        # sheep no renormal
        # newVelocities[0] = currentActions[0]
        # print(newVelocities)


        newPositions = [np.add(currentPositions[i], newVelocities[i])
                        for i in range(numberObjects)]

        for i in range(numberObjects):
            if newPositions[i][0] > self.movingRange[2]:
                newPositions[i][0] = 2 * \
                    self.movingRange[2] - newPositions[i][0]
            if newPositions[i][0] < self.movingRange[0]:
                newPositions[i][0] = 2 * \
                    self.movingRange[0] - newPositions[i][0]
            if newPositions[i][1] > self.movingRange[3]:
                newPositions[i][1] = 2 * \
                    self.movingRange[3] - newPositions[i][1]
            if newPositions[i][1] < self.movingRange[1]:
                newPositions[i][1] = 2 * \
                    self.movingRange[1] - newPositions[i][1]

        newVelocities = [newPositions[i] - currentPositions[i]
                         for i in range(numberObjects)]
        newPhysicalStatesList = [list(newPositions[i]) + list(newVelocities[i])
                         for i in range(numberObjects)]

        newPhysicalStates = pd.DataFrame(
            newPhysicalStatesList, index=currentStates.index, columns=currentStates.columns)

        if self.renderOn:

            currentPositions = [list(currentPositions[i]) for i in range(numberObjects)]

            agentCoordinates = list(map(int, currentPositions[0]))
            wolfCoordinates = list(map(int, currentPositions[1]))
            distractorCoordinates = list(map(int, currentPositions[2]))

            pygame.init()
            screenSize = [self.movingRange[2],self.movingRange[3]]
            screen = pygame.display.set_mode(screenSize)
            circleR = 10
            screen.fill([0,0,0])
            color = [THECOLORS['green'],THECOLORS['red']] + [THECOLORS['blue']] * (numberObjects-2)
            positionList = [agentCoordinates, wolfCoordinates, distractorCoordinates]


            for drawposition in positionList:
                pygame.draw.circle(screen,color[int(positionList.index(drawposition))],drawposition,circleR)
            pygame.display.flip()
            # pygame.time.wait(0.2)

            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()

        return newPhysicalStates

class AccumulateRewards():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action) for state, action in trajectory]
        
        accumulateReward = lambda accumulatedReward, reward: self.decay * accumulatedReward + reward
        accumulatedRewards = np.array([ft.reduce(accumulateReward, reversed(rewards[TimeT: ])) for TimeT in range(len(rewards))])
        return accumulatedRewards

class TrainCriticMonteCarloTensorflow():
    def __init__(self, criticWriter, accumulateRewards):
        self.criticWriter = criticWriter
        self.accumulateRewards = accumulateRewards
    def __call__(self, episode, criticModel):
        mergedEpisode = np.concatenate(episode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.vstack(stateEpisode), np.vstack(actionEpisode)
        
        mergedAccumulatedRewardsEpisode = np.concatenate([self.accumulateRewards(trajectory) for trajectory in episode])
        valueTargetBatch = np.vstack(mergedAccumulatedRewardsEpisode)

        graph = criticModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        valueTarget_ = graph.get_tensor_by_name('inputs/valueTarget_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                          valueTarget_ : valueTargetBatch
                                                                          })
        self.criticWriter.flush()
        return loss, criticModel

class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay, rewardFunction):
        self.criticWriter = criticWriter
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, episode, criticModel):
        
        noLastStateEpisode = [trajectory[ : -1] for trajectory in episode]
        mergedNoLastStateEpisode = np.concatenate(noLastStateEpisode)
        states, actions = list(zip(*mergedNoLastStateEpisode)) 
        
        noFirstStateEpisode = [trajectory[1 : ] for trajectory in episode]
        mergedNoFirstStateEpisode = np.concatenate(noFirstStateEpisode)
        nextStates, nextActions = list(zip(*mergedNoFirstStateEpisode)) 
 
        stateBatch, nextStateBatch = np.vstack(states), np.vstack(nextStates)
        
        graph = criticModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
        nextStateValueBatch = criticModel.run(value_, feed_dict = {state_ : nextStateBatch})
        
        rewardsEpisode = np.array([self.rewardFunction(state, action) for state, action in mergedNoLastStateEpisode])
        rewardBatch = np.vstack(rewardsEpisode)
        valueTargetBatch = rewardBatch + self.decay * nextStateValueBatch

        state_ = graph.get_tensor_by_name('inputs/state_:0')
        valueTarget_ = graph.get_tensor_by_name('inputs/valueTarget_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = criticModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                          valueTarget_ : valueTargetBatch
                                                                          })
        self.criticWriter.flush()
        return loss, criticModel

def approximateValue(stateBatch, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
    valueBatch = criticModel.run(value_, feed_dict = {state_ : stateBatch})
    return valueBatch

class EstimateAdvantageMonteCarlo():
    def __init__(self, accumulateRewards):
        self.accumulateRewards = accumulateRewards
    def __call__(self, episode, critic):
        mergedEpisode = np.concatenate(episode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.vstack(stateEpisode), np.vstack(actionEpisode)
        
        mergedAccumulatedRewardsEpisode = np.concatenate([self.accumulateRewards(trajectory) for trajectory in episode])
        accumulatedRewardsBatch = np.vstack(mergedAccumulatedRewardsEpisode)

        advantageBatch = accumulatedRewardsBatch - critic(stateBatch)
        advantages = np.concatenate(advantageBatch)
        return advantages

class EstimateAdvantageBootstrap():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, episode, critic):
        noLastStateEpisode = [trajectory[ : -1] for trajectory in episode]
        mergedNoLastStateEpisode = np.concatenate(noLastStateEpisode)
        states, actions = list(zip(*mergedNoLastStateEpisode)) 
        
        noFirstStateEpisode = [trajectory[1 : ] for trajectory in episode]
        mergedNoFirstStateEpisode = np.concatenate(noFirstStateEpisode)
        nextStates, nextActions = list(zip(*mergedNoFirstStateEpisode)) 
       
        stateBatch, nextStateBatch = np.vstack(states), np.vstack(nextStates)
        
        rewardsEpisode = np.array([self.rewardFunction(state, action) for state, action in mergedNoLastStateEpisode])
        trajectoryLengthes = [len(trajectory) for trajectory in noLastStateEpisode]
        lastStateIndex = np.cumsum(trajectoryLengthes) - 1
        rewardsEpisode[lastStateIndex] = -20
        rewardBatch = np.vstack(rewardsEpisode)
         
        advantageBatch = rewardBatch + self.decay * critic(nextStateBatch) - critic(stateBatch)
        advantages = np.concatenate(advantageBatch)
        return advantages

class TrainActorMonteCarloTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, episode, advantages, actorModel):
        mergedEpisode = np.concatenate(episode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.vstack(stateEpisode), np.vstack(actionEpisode)

        graph = actorModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0')
        advantages_ = graph.get_tensor_by_name('inputs/advantages_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = actorModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                         action_ : actionBatch,
                                                                         advantages_ : advantages            
                                                                         })
        self.actorWriter.flush()
        return loss, actorModel

class TrainActorBootstrapTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, episode, advantages, actorModel):
        noLastStateEpisode = [trajectory[ : -1] for trajectory in episode]
        mergedEpisode = np.concatenate(noLastStateEpisode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.vstack(stateEpisode), np.vstack(actionEpisode)

        graph = actorModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0')
        advantages_ = graph.get_tensor_by_name('inputs/advantages_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = actorModel.run([loss_, trainOpt_], feed_dict = {state_ : stateBatch,
                                                                         action_ : actionBatch,
                                                                         advantages_ : advantages       
                                                                         })
        self.actorWriter.flush()
        return loss, actorModel

class OfflineAdvantageActorCritic():
    def __init__(self, numTrajectory, maxEpisode):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
    def __call__(self, actorModel, criticModel, approximatePolicy, sampleTrajectory, trainCritic, approximateValue, estimateAdvantage, trainActor):
        for episodeIndex in range(self.maxEpisode):
            actor = lambda state: approximatePolicy(state, actorModel)
            episode = [sampleTrajectory(actor) for index in range(self.numTrajectory)]
            valueLoss, criticModel = trainCritic(episode, criticModel)
            critic = lambda state: approximateValue(state, criticModel)
            advantages = estimateAdvantage(episode, critic)
            policyLoss, actorModel = trainActor(episode, advantages, actorModel)
            print(np.mean([len(episode[index]) for index in range(self.numTrajectory)]))
        return actorModel, criticModel

def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)
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

    numberObjects = 3

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
    distractorUpdateStep = 20

    numActionSpace = 2
    numStateSpace = numberObjects * 4 + (numberObjects - 1) * len(assumeWolfPrecisionList)

    actionLow = -2
    actionHigh = 2
    actionRatio = (actionHigh - actionLow) / 2.

    envModelName = 'inverted_pendulum'
    renderOn = True
    maxTimeStep = 200
    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    rewardDecay = 0.99

    numTrajectory = 200 
    maxEpisode = 1000

    learningRateActor = 0.0001
    learningRateCritic = 0.001
 
    savePathActor = 'data/tmpModelActor.ckpt'
    savePathCritic = 'data/tmpModelCritic.ckpt'
    
    actorGraph = tf.Graph()
    with actorGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            action_ = tf.placeholder(tf.float32, [None, numActionSpace], name="action_")
            advantages_ = tf.placeholder(tf.float32, [None, ], name="advantages_")

        with tf.name_scope("hidden"):
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)
            fullyConnected2_ = tf.layers.dense(inputs = fullyConnected1_, units = 20, activation = tf.nn.relu)
            actionMean_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.tanh)
            actionVariance_ = tf.layers.dense(inputs = fullyConnected2_, units = numActionSpace, activation = tf.nn.softplus)

        with tf.name_scope("outputs"):        
            actionDistribution_ = tfp.distributions.MultivariateNormalDiag(actionMean_ * actionRatio, actionVariance_ + 1e-8, name = 'actionDistribution_')
            actionSample_ = tf.clip_by_value(actionDistribution_.sample(), actionLow, actionHigh, name = 'actionSample_')
            negLogProb_ = - actionDistribution_.log_prob(action_, name = 'negLogProb_')
            loss_ = tf.reduce_sum(tf.multiply(negLogProb_, advantages_), name = 'loss_')
        actorLossSummary = tf.summary.scalar("ActorLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateActor, name = 'adamOpt_').minimize(loss_)

        actorInit = tf.global_variables_initializer()
        
        actorSummary = tf.summary.merge_all()
        actorSaver = tf.train.Saver(tf.global_variables())

    actorWriter = tf.summary.FileWriter('tensorBoard/actor', graph = actorGraph)
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")

        with tf.name_scope("hidden"):
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = 30, activation = tf.nn.relu)

        with tf.name_scope("outputs"):        
            value_ = tf.layers.dense(inputs = fullyConnected1_, units = 1, activation = None, name = 'value_')
            diff_ = tf.subtract(valueTarget_, value_, name = 'diff_')
            loss_ = tf.reduce_mean(tf.square(diff_), name = 'loss_')
        criticLossSummary = tf.summary.scalar("CriticLoss", loss_)

        with tf.name_scope("train"):
            trainOpt_ = tf.train.AdamOptimizer(learningRateCritic, name = 'adamOpt_').minimize(loss_)

        criticInit = tf.global_variables_initializer()
        
        criticSummary = tf.summary.merge_all()
        criticSaver = tf.train.Saver(tf.global_variables())
    
    criticWriter = tf.summary.FileWriter('tensorBoard/critic', graph = criticGraph)
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)    
     
    #transitionFunction = env.TransitionFunction(envModelName, renderOn)
    #isTerminal = env.IsTerminal(maxQPos)
    #reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    computePrecisionAndDecay = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
    switchAttention = Attention.AttentionSwitch(attentionLimitation)
    updateBelief = BeliefUpdate.BeliefUpdateWithAttention(computePrecisionAndDecay, switchAttention, attentionSwitchFrequency, sheepIdentity)

    initialPosition = InitialPosition.InitialPosition(movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
    reset = Reset(numberObjects, initialPosition, assumeWolfPrecisionList,attentionLimitation)

    wolfPolicy = PreparePolicy.WolfPolicy(sheepIdentity, wolfIdentity, speedList[wolfIdentity])
    distractorPolicy = PreparePolicy.DistractorPolicy(distractorIdentity, distractorPrecision, speedList[distractorIdentity])

    # wolfPrecision = random.choice(assumeWolfPrecisionList)
    wolfPrecision = 50

    transitionFunction = TransitionWithBelief(movingRange, speedList, numberObjects, assumeWolfPrecisionList, updateBelief, renderOn=True)
    sampleTrajectory = SampleTrajectory(maxTimeStep, numberObjects, transitionFunction, isTerminal, reset, distractorUpdateStep, wolfPolicy, distractorPolicy, wolfPrecision)

    rewardFunction = RewardFunction(aliveBouns)
    accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

    trainCritic = TrainCriticMonteCarloTensorflow(criticWriter, accumulateRewards)
    estimateAdvantage = EstimateAdvantageMonteCarlo(accumulateRewards)
    trainActor = TrainActorMonteCarloTensorflow(actorWriter) 
    
    #trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, rewardFunction)
    #estimateAdvantage = EstimateAdvantageBootstrap(rewardDecay, rewardFunction)
    #trainActor = TrainActorBootstrapTensorflow(actorWriter) 

    actorCritic = OfflineAdvantageActorCritic(numTrajectory, maxEpisode)

    trainedActorModel, trainedCriticModel = actorCritic(actorModel, criticModel, approximatePolicy, sampleTrajectory, trainCritic,
            approximateValue, estimateAdvantage, trainActor)

    with actorModel.as_default():
        actorSaver.save(trainedActorModel, savePathActor)
    with criticModel.as_default():
        criticSaver.save(trainedCriticModel, savePathCritic)

    # transitionPlay = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = True)
    # samplePlay = SampleTrajectory(maxTimeStep, transitionPlay, isTerminal, reset)
    # actor = lambda state: approximatePolicy(state, trainedActorModel)
    # playEpisode = [samplePlay(actor) for index in range(5)]
    # print(np.mean([len(playEpisode[index]) for index in range(5)]))

if __name__ == "__main__":
    main()
