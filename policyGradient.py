import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import functools as ft
#import env
# import cartpole_env
# import reward
import dataSave
import singleWolfEnv as env
import InitialPosition
import PreparePolicy


def approximatePolicy(stateBatch, model):
    graph = model.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    actionSample_ = graph.get_tensor_by_name('outputs/actionSample_:0')
    actionBatch = model.run(actionSample_, feed_dict={state_: stateBatch})
    return actionBatch


class RewardFunctionTerminalPenalty():
    def __init__(self, aliveBouns, deathPenalty, isTerminal, shapeReward):
        self.aliveBouns = aliveBouns
        self.deathPenalty = deathPenalty
        self.isTerminal = isTerminal
        self.shapeReward = shapeReward

    def __call__(self, state, action):
        reward = self.aliveBouns
        if self.isTerminal(state):
            reward = self.deathPenalty

        if self.shapeReward:
            agentState = state[:4]
            wolfState = state[4:8]
            distance = env.l2Norm(agentState[:2], wolfState[:2])
            distanceReward = distance * 0.01
            reward += distanceReward
        return reward


class SampleTrajectory():
    def __init__(self, maxTimeStep, transitionFunction, isTerminal, reset):
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset

    def __call__(self, policy):
        oldState = self.reset()
        trajectory = []

        for time in range(self.maxTimeStep):
            oldStateBatch = oldState.reshape(1, -1)
            actionBatch = policy(oldStateBatch)
            action = actionBatch[0]
            # actionBatch shape: batch * action Dimension; only need action Dimention
            newState = self.transitionFunction(oldState, action)
            trajectory.append((oldState, action))

            terminal = self.isTerminal(oldState)
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

    def __call__(self, episode, episodeIndex, normalizedAccumulatedRewardsEpisode, model):
        mergedEpisode = np.concatenate(episode)
        numBatch = len(mergedEpisode)
        stateEpisode, actionEpisode = list(zip(*mergedEpisode))
        stateBatch, actionBatch = np.array(stateEpisode).reshape(
            numBatch, -1), np.array(actionEpisode).reshape(numBatch, -1)
        mergedAccumulatedRewardsEpisode = np.concatenate(
            normalizedAccumulatedRewardsEpisode)

        graph = model.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        action_ = graph.get_tensor_by_name('inputs/action_:0')
        accumulatedRewards_ = graph.get_tensor_by_name(
            'inputs/accumulatedRewards_:0')
        loss_ = graph.get_tensor_by_name('outputs/loss_:0')
        trainOpt_ = graph.get_operation_by_name('train/adamOpt_')
        lossSummary = graph.get_tensor_by_name('outputs/Loss:0')
        loss, summary, trainOpt = model.run([loss_, lossSummary, trainOpt_], feed_dict={state_: stateBatch,
                                                                                        action_: actionBatch,
                                                                                        accumulatedRewards_: mergedAccumulatedRewardsEpisode
                                                                                        })
        self.summaryWriter.add_summary(summary, episodeIndex)
        self.summaryWriter.flush()
        return loss, model


class PolicyGradient():
    def __init__(self, numTrajectory, maxEpisode):
        self.numTrajectory = numTrajectory
        self.maxEpisode = maxEpisode

    def __call__(self, model, approximatePolicy, sampleTrajectory, accumulateRewards, train):
        for episodeIndex in range(self.maxEpisode):
            def policy(state): return approximatePolicy(state, model)
            episode = [sampleTrajectory(policy)
                       for index in range(self.numTrajectory)]
            normalizedAccumulatedRewardsEpisode = [
                normalize(accumulateRewards(trajectory)) for trajectory in episode]
            loss, model = train(episode, episodeIndex,
                                normalizedAccumulatedRewardsEpisode, model)
            print(np.mean([len(episode[index])
                           for index in range(self.numTrajectory)]))
        return model


def main():
    speedList = [8, 4, 4, 4, 4, 4]
    movingRange = [0, 0, 364, 364]
    numberObjects = 2
    sheepIdentity = 0
    wolfIdentity = 1
    numActionSpace = 2
    actionLow = -2
    actionHigh = 2
    actionRatio = (actionHigh - actionLow) / 2.

    renderOn = True
    shapeReward = True
    numStateSpace = numberObjects * 4

    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001

    aliveBouns = 1
    deathPenalty = -20
    rewardDecay = 1
    maxDistanceToFixation = movingRange[3]
    minDistanceEachOther = 50
    maxDistanceEachOther = 180
    minDistanceWolfSheep = 120

    maxTimeStep = 1000
    numTrajectory = 100
    maxEpisode = 1000

    learningRate = 0.01
    summaryPath = 'tensorBoard/policyGradientContinuous'

    savePath = 'data/tmpModelGaussian.ckpt'

    with tf.name_scope("inputs"):
        state_ = tf.placeholder(
            tf.float32, [None, numStateSpace], name="state_")
        action_ = tf.placeholder(
            tf.float32, [None, numActionSpace], name="action_")
        accumulatedRewards_ = tf.placeholder(
            tf.float32, [None, ], name="accumulatedRewards_")

    with tf.name_scope("hidden"):
        fullyConnected1_ = tf.layers.dense(
            inputs=state_, units=30, activation=tf.nn.relu)
        fullyConnected2_ = tf.layers.dense(
            inputs=fullyConnected1_, units=20, activation=tf.nn.relu)
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
    model = tf.Session()
    model.run(tf.global_variables_initializer())
    summaryWriter = tf.summary.FileWriter(summaryPath, graph=model.graph)

    """
    transitionFunction = env.TransitionFunction(envModelName, renderOn)
    isTerminal = env.IsTerminal(maxQPos)
    reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    """
    isTerminal = env.isTerminal
    wolfPolicy = PreparePolicy.WolfPolicy(
        sheepIdentity, wolfIdentity, speedList[wolfIdentity])
    wolfPrecision = 50

    initialPosition = InitialPosition.InitialPosition(
        movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
    transitionFunction = env.Transition(
        movingRange, speedList, numberObjects, wolfPolicy, wolfPrecision, renderOn)
    reset = env.Reset(numberObjects, initialPosition, movingRange, speedList)

    sampleTrajectory = SampleTrajectory(
        maxTimeStep, transitionFunction, isTerminal, reset)

    rewardFunction = RewardFunctionTerminalPenalty(
        aliveBouns, deathPenalty, isTerminal, shapeReward, movingRange)
    #rewardFunction = reward.CartpoleRewardFunction(aliveBouns)
    accumulateRewards = AccumulateRewards(rewardDecay, rewardFunction)

    train = TrainTensorflow(summaryWriter)

    policyGradient = PolicyGradient(numTrajectory, maxEpisode)

    trainedModel = policyGradient(
        model, approximatePolicy, sampleTrajectory, accumulateRewards, train)

    saveModel = dataSave.SaveModel(savePath)
    modelSave = saveModel(model)

    # transitionPlay = cartpole_env.Cartpole_continuous_action_transition_function(renderOn = True)
    # samplePlay = SampleTrajectory(maxTimeStep, transitionPlay, isTerminal, reset)
    # policy = lambda state: approximatePolicy(state, model)
    # playEpisode = [samplePlay(policy) for index in range(5)]
    # print(np.mean([len(playEpisode[index]) for index in range(5)]))
if __name__ == "__main__":
    main()
