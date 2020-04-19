import tensorflow as tf
import tensorflow_probability as tfp
# import edward as ed # for linux
import numpy as np
import pandas as pd
import functools as ft
# import env
# import cartpole_env
import random
import dataSave

import Attention
import BeliefUpdate
import InitialPosition
import PreparePolicy
import env

import pygame
from pygame.color import THECOLORS
from pygame.locals import *

from multiprocessing import Pool

class RewardFunction():
    def __init__(self, aliveBouns, deathPenalty):
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty

    def __call__(self, oldState, action):
        if isTerminal(oldState):
            penalty = self.deathPenalty
            return penalty
        reward = self.aliveBouns
        return reward

class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset, numStateSpace):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
        self.numStateSpace = numStateSpace

    def __call__(self, policy):
        oldState, self.transitionFunction = self.reset()
        trajectory = []

    
        for time in range(self.maxTimeStep):
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = policy(oldStateBatch)
            action = actionBatch[0]
            # actionBatch shape: batch * action Dimension; only need action Dimention
            trajectory.append((oldState, action))

            newState = self.transitionFunction(oldState, action)
            terminal = self.isTerminal(oldState)

            beliefACC = np.sum(oldState[:self.numStateSpace,].flatten()[4*3::5])
            print('beliefACC:',beliefACC)
            if terminal:
                break

            oldState = newState
        return trajectory

class AccumulateRewards():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction

    def __call__(self, trajectory):
        rewards = [self.rewardFunction(state, action)
                   for state, action in trajectory]

        def accumulateReward(
            accumulatedReward, reward): return self.decay * accumulatedReward + reward
        accumulatedRewards = np.array([ft.reduce(accumulateReward, reversed(
            rewards[TimeT:])) for TimeT in range(len(rewards))])
        return accumulatedRewards


def normalize(accumulatedRewards):
    normalizedAccumulatedRewards = (
        accumulatedRewards - np.mean(accumulatedRewards)) / np.std(accumulatedRewards)
    return normalizedAccumulatedRewards


class TrainTensorflow():
    def __init__(self, summaryWriter):
        self.summaryWriter = summaryWriter

    def __call__(self, episode, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.vstack(
            stateEpisode), np.vstack(actionEpisode)
        mergedAccumulatedRewardsEpisode = np.concatenate(
            normalizedAccumulatedRewardsEpisode)

        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0')
        accumulatedRewards_ = graph.get_tensor_by_name(
            'inputs/accumulatedRewards_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        loss, trainOpt = model.run([loss_, trainOpt_], feed_dict={state_: np.vstack(stateBatch),
                                                                  action_: np.vstack(actionBatch),
                                                                  accumulatedRewards_: mergedAccumulatedRewardsEpisode
                                                                  })
        self.summaryWriter.flush()
        return loss, model

def approximatePolicy(stateBatch, model):
    graph = model.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    actionSample_ = graph.get_tensor_by_name('outputs/actionSample_:0')
    actionBatch = model.run(actionSample_, feed_dict={state_: stateBatch})
    return actionBatch
    
class PolicyGradient():
    def __init__(self, numTrajectory, maxEpisode, savePath):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode
        self.savePath = savePath

    def __call__(self, model, approximatePolicy, sampleTrajectory, accumulateRewards, train, saver):
        for episodeIndex in range(self.maxEpisode):
            def policy(state): return approximatePolicy(state, model)
            episode = [sampleTrajectory(policy)
                       for index in range(self.numTrajectory)]
            normalizedAccumulatedRewardsEpisode = [
                normalize(accumulateRewards(trajectory)) for trajectory in episode]
            loss, model = train(
                episode, normalizedAccumulatedRewardsEpisode, model)

            save_path = saver.save(model, self.savePath)
            score = np.mean([len(episode[index])
                           for index in range(self.numTrajectory)])
            print("Model saved in path: %s" % save_path)
            print("episodeIndex: ", episodeIndex, "loss: ", loss)
            print("episodeIndex: ", episodeIndex, "score: ", score)
        return model




def main():
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

    actionLow = - 2
    actionHigh = 2

    actionRatio = (actionHigh - actionLow) / 2.

    renderOn = True
    maxTimeStep = 1000
    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    deathPenalty = -20

    rewardDecay = 1

    numTrajectory = 100
    maxEpisode = 1000

    learningRate = 0.001
    summaryPath = 'tensorBoard/1'

    savePath = 'data/model.ckpt'
    useSavedModel = 0
   
    with tf.name_scope("inputs"):
        state_ = tf.placeholder(
            tf.float32, [None, numStateSpace], name="state_")
        action_ = tf.placeholder(
            tf.float32, [None, numActionSpace], name="action_")
        accumulatedRewards_ = tf.placeholder(
            tf.float32, [None, ], name="accumulatedRewards_")

    with tf.name_scope("hidden"):
        fullyConnected1_ = tf.layers.dense(
            inputs=state_, units=64, activation=tf.nn.relu)
        fullyConnected2_ = tf.layers.dense(
            inputs=fullyConnected1_, units=64, activation=tf.nn.relu)
        actionMean_ = tf.layers.dense(
            inputs=fullyConnected2_, units=numActionSpace, activation=tf.nn.tanh)
        actionVariance_ = tf.layers.dense(
            inputs=fullyConnected2_, units=numActionSpace, activation=tf.nn.softplus)

    with tf.name_scope("outputs"):
        actionDistribution_ = tfp.distributions.MultivariateNormalDiag(
            actionMean_ * actionRatio, actionVariance_ + 1e-8, name='actionDistribution_')
        actionSample_ = tf.clip_by_value(
            actionDistribution_.sample(), actionLow, actionHigh, name='actionSample_')
        negLogProb_ = - \
            actionDistribution_.log_prob(action_, name='negLogProb_')
        loss_ = tf.reduce_sum(tf.multiply(
            negLogProb_, accumulatedRewards_), name='loss_')
        tf.summary.scalar("Loss", loss_)

    with tf.name_scope("train"):
        trainOpt_ = tf.train.AdamOptimizer(
            learningRate, name='adamOpt_').minimize(loss_)

    mergedSummary = tf.summary.merge_all()
    saver = tf.train.Saver()

    model = tf.Session()

    if useSavedModel:
        saver.restore(model, savePath)
    else:
        model.run(tf.global_variables_initializer())

    summaryWriter = tf.summary.FileWriter(summaryPath, graph=model.graph)

    computePrecisionAndDecay = Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
    switchAttention = Attention.AttentionSwitch(attentionLimitation)
    updateBelief = BeliefUpdate.BeliefUpdateWithAttention(computePrecisionAndDecay, switchAttention, attentionSwitchFrequency, sheepIdentity)
    initialPosition = InitialPosition.InitialPosition(movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
    wolfPolicy = PreparePolicy.WolfPolicy(sheepIdentity, wolfIdentity, speedList[wolfIdentity])
    distractorPolicy = PreparePolicy.DistractorPolicy(distractorIdentity, distractorPrecision, speedList[distractorIdentity])

    # wolfPrecision = random.choice(assumeWolfPrecisionList)
    wolfPrecision = 50

    transitionFunction = env.TransitionWithBelief(movingRange, speedList, numberObjects, assumeWolfPrecisionList, attentionLimitation, updateBelief, wolfPolicy,wolfPrecision,distractorPolicy,distractorUpdateStep,renderOn)
    reset = env.Reset(numberObjects, initialPosition, assumeWolfPrecisionList, attentionLimitation, movingRange, speedList, updateBelief,wolfPolicy,wolfPrecision,distractorPolicy,distractorUpdateStep,renderOn)
    
    sampleTrajectory = SampleTrajectory(maxTimeStep, transitionFunction, env.isTerminal, reset, numStateSpace)

    rewardFunction = RewardFunction(aliveBouns, deathPenalty)
    accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

    train = TrainTensorflow(summaryWriter)
        
    policyGradient = PolicyGradient(numTrajectory, maxEpisode, savePath)
    trainedModel = policyGradient(
        model, approximatePolicy, sampleTrajectory, accumulateRewards, train, saver)

    saveModel = dataSave.SaveModel(savePath)
    modelSave = saveModel(model)

    # transitionPlay = Transition(movingRange, speedList, sheepIdentity, renderOn=True)

    # samplePlay = SampleTrajectory(
    #     maxTimeStep, transitionPlay, isTerminal, reset)

    # def policy(state): return approximatePolicy(state, model)
    # playEpisode = [samplePlay(policy) for index in range(5)]
    # print(np.mean([len(playEpisode[index]) for index in range(5)]))


if __name__ == "__main__":
    main()
