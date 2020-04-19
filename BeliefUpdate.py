
import pandas as pd
import numpy as np
import scipy.stats as stats
import math
import Attention


def computeDeviationFromTrajectory(vector1, vector2):
    def calVectorNorm(vector):
        return np.power(np.power(vector, 2).sum(axis=1), 0.5)
    innerProduct = np.dot(vector1, vector2.T).diagonal()
    angle = np.arccos(
        innerProduct / (calVectorNorm(vector1) * calVectorNorm(vector2)))
    return angle


def initiateBeliefDF(objectsNumber, assumeWolfPrecisionList):
    multiIndex = pd.MultiIndex.from_product([assumeWolfPrecisionList, range(
        1, objectsNumber)], names=['assumeChasingPrecision', 'Identity'])
    initialProbability = [1.0 / len(multiIndex)] * len(multiIndex)
    beliefDF = pd.DataFrame(
        initialProbability, index=multiIndex, columns=['p'])
    return beliefDF


def computeDeviationAngleDF(oldStates, currentStates, sheepIdentity):
    oldStatesSelfDF = oldStates.loc[sheepIdentity][[
        'positionX', 'positionY', 'velocityX', 'velocityY']]
    oldStatesOthersDF = oldStates.loc[1:][[
        'positionX', 'positionY', 'velocityX', 'velocityY']]
    currentStatesSelfDF = currentStates.loc[sheepIdentity]
    currentStatesOthersDF = currentStates.loc[1:]
    assumeDirectionDF = (
        oldStatesSelfDF - oldStatesOthersDF).loc[:][['positionX', 'positionY']]
    observeDirectionDF = currentStatesOthersDF.loc[:][[
        'velocityX', 'velocityY']]
    deviationAngle = computeDeviationFromTrajectory(
        assumeDirectionDF, observeDirectionDF)
    deviationAngleDF = pd.DataFrame(
        deviationAngle.values, index=assumeDirectionDF.index, columns=['chasingDeviation'])
    return deviationAngleDF


def createHypothesisInformationDF(deviationAngleDF, oldBelief):
    objectIdentity = list(deviationAngleDF.index)
    assumeWolfPrecisionList = list(oldBelief.groupby(
        'assumeChasingPrecision').sum().index)
    hypothesisIndex = oldBelief.index
    hypothesisInformation = pd.DataFrame(list(deviationAngleDF.values) * len(
        assumeWolfPrecisionList), index=hypothesisIndex, columns=['chasingDeviation'])
    hypothesisInformation['pPrior'] = list(oldBelief['p'].values)
    return hypothesisInformation


def computeLikelihood(deviationAngle, assumePrecision):
    pLikelihood = stats.vonmises.pdf(
        deviationAngle, assumePrecision) * 2 * math.pi
    return pLikelihood


class BeliefUpdate():
    def __init__(self, sheepIdentity):
        self.sheepIdentity = sheepIdentity

    def __call__(self, oldBelief, oldStates, currentStates):
        deviationAngleDF = computeDeviationAngleDF(
            oldStates, currentStates, self.sheepIdentity)
        hypothesisInformation = createHypothesisInformationDF(
            deviationAngleDF, oldBelief)
        hypothesisInformation['pLikelihood'] = computeLikelihood(hypothesisInformation['chasingDeviation'].values, hypothesisInformation.index.get_level_values('assumeChasingPrecision'))
        hypothesisInformation['p'] = hypothesisInformation['pPrior'] * hypothesisInformation['pLikelihood']
        hypothesisInformation['p'] = hypothesisInformation['p'] / \
            hypothesisInformation['p'].sum()
        currentBelief = oldBelief.copy()
        currentBelief['p'] = hypothesisInformation['p']
        return currentBelief


def renormalVector(rawVector, targetLength):
    rawLength = np.power(np.power(rawVector, 2).sum(), 0.5)
    changeRate = np.divide(targetLength, rawLength)
    return np.multiply(rawVector, changeRate)


class Transition():
    def __init__(self, movingRange, speedList):
        self.movingRange = movingRange
        self.speedList = speedList

    def __call__(self, currentStates, currentActions):
        currentPositions = currentStates.loc[:][[
            'positionX', 'positionY']].values
        currentVelocities = currentStates.loc[:][[
            'velocityX', 'velocityY']].values
        numberObjects = len(currentStates.index)

        newVelocities = [renormalVector(np.add(currentVelocities[i], np.divide(
            currentActions[i], 2.0)), self.speedList[i]) for i in range(numberObjects)]
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
        newStatesList = [list(newPositions[i]) + list(newVelocities[i])
                         for i in range(numberObjects)]
        newStates = pd.DataFrame(
            newStatesList, index=currentStates.index, columns=currentStates.columns)
        return newStates


if __name__ == "__main__":
    import datetime
    time0 = datetime.datetime.now()
    objectsNumber = 3
    statesList = [[10, 10, 0, 0], [10, 5, 0, 0], [9, 9, 0, 0]]
    oldStates = pd.DataFrame(statesList, index=[0, 1, 2], columns=[
                             'positionX', 'positionY', 'velocityX', 'velocityY'])
    speedList = [5, 3, 3]
    currentActions = [[0, 3], [0, 3], [-3, -3]]
    movingRange = [0, 0, 15, 15]
    assumeWolfPrecisionList = [50]  # [50, 1.3]
    sheepIdentity = 0
    oldBelief = initiateBeliefDF(objectsNumber, assumeWolfPrecisionList)

    # print('initialParameters', datetime.datetime.now() - time0)
    transState = Transition(movingRange, speedList)
    updateBelief = BeliefUpdate(sheepIdentity)
    # print('initialFunctions', datetime.datetime.now() - time0)
    currentStates = transState(oldStates, currentActions)
    # print('updateState', datetime.datetime.now() - time0)
    currentBelief = updateBelief(
        oldBelief, oldStates, currentStates)
    # print('updateBelief', datetime.datetime.now() - time0)
    print(oldBelief)
    print(currentStates)
    print(currentBelief)
