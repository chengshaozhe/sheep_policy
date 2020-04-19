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

def approximatePolicy(stateBatch, actorModel):
    graph = actorModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    actionSample_ = graph.get_tensor_by_name('outputs/actionSample_:0')
    actionBatch = actorModel.run(actionSample_, feed_dict = {state_ : stateBatch})
    return actionBatch

def approximateValue(stateBatch, criticModel):
    graph = criticModel.graph
    state_ = graph.get_tensor_by_name('inputs/state_:0')
    value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
    valueBatch = criticModel.run(value_, feed_dict = {state_ : stateBatch})
    return valueBatch

class TrainCriticBootstrapTensorflow():
    def __init__(self, criticWriter, decay, rewardFunction):
        self.criticWriter = criticWriter
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, state, action, nextState, criticModel):
        
        stateBatch, actionBatch, nextStateBatch = state.reshape(1, -1), action.reshape(1, -1), nextState.reshape(1, -1)
        rewardBatch = np.array([self.rewardFunction(state, action) for state, action in zip(stateBatch, actionBatch)]) 

        graph = criticModel.graph
        state_ = graph.get_tensor_by_name('inputs/state_:0')
        value_ = graph.get_tensor_by_name('outputs/value_/BiasAdd:0')
        nextStateValueBatch = criticModel.run(value_, feed_dict = {state_ : nextStateBatch})
        
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

class EstimateAdvantageBootstrap():
    def __init__(self, decay, rewardFunction):
        self.decay = decay
        self.rewardFunction = rewardFunction
    def __call__(self, state, action, nextState, critic):
        
        stateBatch, actionBatch, nextStateBatch = state.reshape(1, -1), action.reshape(1, -1), nextState.reshape(1, -1)
        rewardBatch = np.array([self.rewardFunction(state, action) for state, action in zip(stateBatch, actionBatch)]) 
        advantageBatch = rewardBatch + self.decay * critic(nextStateBatch) - critic(stateBatch)
        advantages = np.concatenate(advantageBatch)
        return advantages

class TrainActorTensorflow():
    def __init__(self, actorWriter):
        self.actorWriter = actorWriter
    def __call__(self, state, action, advantages, actorModel):
        stateBatch, actionBatch = state.reshape(1, -1), action.reshape(1, -1)
        
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

class OnlineAdvantageActorCritic():
    def __init__(self, maxEpisode, maxTimeStep, transitionFunction, isTerminal, reset):
        self.maxEpisode = maxEpisode
        self.maxTimeStep = maxTimeStep
        self.transitionFunction = transitionFunction
        self.isTerminal = isTerminal
        self.reset = reset
    def __call__(self, actorModel, criticModel, approximatePolicy, trainCritic, approximateValue, estimateAdvantage, trainActor):
        for episodeIndex in range(self.maxEpisode):
            oldState = self.reset()
            for timeStepIndex in range(self.maxTimeStep):
                actor = lambda state: approximatePolicy(state, actorModel)
                actionBatch = actor(oldState.reshape(1, -1))
                action = actionBatch[0]
                newState = self.transitionFunction(oldState, action)
                valueLoss, criticModel = trainCritic(oldState, action, newState, criticModel)
                critic = lambda state: approximateValue(state, criticModel)
                advantage = estimateAdvantage(oldState, action, newState, critic)
                policyLoss, actorModel = trainActor(oldState, action, advantage, actorModel)
                if self.isTerminal(oldState):
                    break
                oldState = newState
            print(timeStepIndex)
        return actorModel, criticModel

def main():
    #tf.set_random_seed(123)
    #np.random.seed(123)

    speedList = [8,4,4,4,4,4]
    movingRange = [0,0,364,364]
    numberObjects = 2
    sheepIdentity = 0
    wolfIdentity = 1
    numStateSpace = numberObjects * 4
    numActionSpace = 2

    actionLow = -2
    actionHigh = 2
    actionRatio = (actionHigh - actionLow) / 2.

    renderOn = True
    shapeReward = True
    isWallPunish = True

    maxDistanceToFixation = movingRange[3]
    minDistanceEachOther = 50
    maxDistanceEachOther = 180
    minDistanceWolfSheep = 120
    
    aliveBouns = 1
    deathPenalty = -20
    rewardDecay = 0.99

    maxQPos = 0.2
    qPosInitNoise = 0.001
    qVelInitNoise = 0.001


    maxTimeStep = 1000
    maxEpisode = 10000

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

    actorWriter = tf.summary.FileWriter('tensorBoard/actorOnlineA2C', graph = actorGraph)
    actorModel = tf.Session(graph = actorGraph)
    actorModel.run(actorInit)    
    
    criticGraph = tf.Graph()
    with criticGraph.as_default():
        with tf.name_scope("inputs"):
            state_ = tf.placeholder(tf.float32, [None, numStateSpace], name="state_")
            valueTarget_ = tf.placeholder(tf.float32, [None, 1], name="valueTarget_")

        with tf.name_scope("hidden1"):
            fullyConnected1_ = tf.layers.dense(inputs = state_, units = 100, activation = tf.nn.relu)

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
    
    criticWriter = tf.summary.FileWriter('tensorBoard/criticOnlineA2C', graph = criticGraph)
    criticModel = tf.Session(graph = criticGraph)
    criticModel.run(criticInit)    
     
    #transitionFunction = env.TransitionFunction(envModelName, renderOn)
    #isTerminal = env.IsTerminal(maxQPos)
    #reset = env.Reset(envModelName, qPosInitNoise, qVelInitNoise)
    isTerminal = env.isTerminal
    wolfPolicy = PreparePolicy.WolfPolicy(sheepIdentity, wolfIdentity, speedList[wolfIdentity])
    wolfPrecision = 50

    initialPosition = InitialPosition.InitialPosition(movingRange, maxDistanceToFixation, minDistanceEachOther, maxDistanceEachOther, minDistanceWolfSheep)
    transitionFunction = env.Transition(movingRange, speedList, numberObjects, wolfPolicy, wolfPrecision, renderOn)
    reset = env.Reset(numberObjects, initialPosition, movingRange, speedList)
   
    rewardFunction = RewardFunctionTerminalPenalty(aliveBouns, deathPenalty, isTerminal, shapeReward)
    #rewardFunction = reward.CartpoleRewardFunction(aliveBouns)
 
    trainCritic = TrainCriticBootstrapTensorflow(criticWriter, rewardDecay, rewardFunction)
    estimateAdvantage = EstimateAdvantageBootstrap(rewardDecay, rewardFunction)
    
    trainActor = TrainActorTensorflow(actorWriter) 

    actorCritic = OnlineAdvantageActorCritic(maxEpisode, maxTimeStep, transitionFunction, isTerminal, reset)

    trainedActorModel, trainedCriticModel = actorCritic(actorModel, criticModel, approximatePolicy, trainCritic,
            approximateValue, estimateAdvantage, trainActor)

    with actorModel.as_default():
        actorSaver.save(trainedActorModel, savePathActor)
    with criticModel.as_default():
        criticSaver.save(trainedCriticModel, savePathCritic)

if __name__ == "__main__":
    main()

