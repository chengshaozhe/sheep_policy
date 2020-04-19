import numpy as np
import pandas as pd
import random
import dataSave

import BeliefUpdate
import PreparePolicy
import InitialPosition
import Attention

import pygame
from pygame.color import THECOLORS
from pygame.locals import *


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

def beliefArrayToDataFrame(numberObjects, oldBelief, assumeWolfPrecisionList):
    multiIndex = pd.MultiIndex.from_product([assumeWolfPrecisionList, range(
        1, numberObjects)], names=['assumeChasingPrecision', 'Identity'])
    oldBeliefList = list(oldBelief)
    beliefDF = pd.DataFrame(
        oldBeliefList, index=multiIndex, columns=['p'])
    return beliefDF

def positionArraytToDataFrame(numberObjects, oldPosition):
    oldPositionList = oldPosition.tolist()
    oldPositionList = [oldPositionList[i:i+4] for i in range(0,len(oldPositionList),4)]
    oldPositionDF = pd.DataFrame(oldPositionList,index=list(range(numberObjects)),
    columns=['positionX','positionY','velocityX','velocityY'])
    return oldPositionDF

def attentionArraytToDataFrame(oldAttentionStatus, oldBelief, attentionLimitation):
    attentionStatusList = oldAttentionStatus.tolist()
    attentionStatusDF = pd.DataFrame(
        attentionStatusList, index=oldBelief.index, columns=['attentionStatus'])
    return attentionStatusDF

def renormalVector(rawVector, targetLength):
    rawLength = np.power(np.power(rawVector, 2).sum(), 0.5)
    changeRate = np.divide(targetLength, rawLength)
    return np.multiply(rawVector, changeRate)

class Reset():
    def __init__(self, numberObjects, initialPosition, assumeWolfPrecisionList, attentionLimitation, movingRange, speedList, updateBelief,wolfPolicy,wolfPrecision,distractorPolicy,distractorUpdateStep,renderOn):
        self.numberObjects = numberObjects
        self.initialPosition = initialPosition
        self.assumeWolfPrecisionList = assumeWolfPrecisionList
        self.attentionLimitation = attentionLimitation
        self.movingRange = movingRange
        self.speedList = speedList 
        self.updateBelief = updateBelief
        self.renderOn = renderOn
        self.wolfPolicy = wolfPolicy
        self.wolfPrecision = wolfPrecision
        self.distractorPolicy = distractorPolicy
        self.distractorUpdateStep = distractorUpdateStep

    def __call__(self):
        initPositionList = self.initialPosition(self.numberObjects)
        statesList = []
        initVelocity = [0,0]
        for initPosition in initPositionList:
            statesList.append(initPosition + initVelocity)

        initState = pd.DataFrame(statesList,index=list(range(self.numberObjects)),
            columns=['positionX','positionY','velocityX','velocityY'])
        initBelief = BeliefUpdate.initiateBeliefDF(self.numberObjects, self.assumeWolfPrecisionList)
        initAttentionStatus = BeliefUpdate.initiateAttentionStatus(initBelief, self.attentionLimitation)

        initPhysicalState = np.asarray(initState).flatten()
        initBelief = np.asarray(initBelief).flatten()

        initState = np.concatenate((initPhysicalState, initBelief))

        initTransitionFunction = TransitionWithBelief(self.movingRange, self.speedList, self.numberObjects, self.assumeWolfPrecisionList, self.attentionLimitation,initAttentionStatus, self.updateBelief, self.wolfPolicy, self.wolfPrecision, self.distractorPolicy, self.distractorUpdateStep,self.renderOn)
        return initState, initTransitionFunction

class TransitionWithBelief():
    def __init__(self, movingRange, speedList, numberObjects, assumeWolfPrecisionList, attentionLimitation, initAttentionStatus, updateBelief, wolfPolicy, wolfPrecision, distractorPolicy, distractorUpdateStep, renderOn=False):
        self.movingRange = movingRange
        self.speedList = speedList
        self.renderOn = renderOn
        self.numberObjects = numberObjects
        self.assumeWolfPrecisionList = assumeWolfPrecisionList
        self.updateBelief = updateBelief
        self.attentionLimitation = attentionLimitation
        self.time = 0
        self.wolfPolicy = wolfPolicy
        self.wolfPrecision = wolfPrecision
        self.distractorPolicy = distractorPolicy
        self.distractorUpdateStep = distractorUpdateStep
        self.oldAttentionStatus =  initAttentionStatus

    def __call__(self, oldState, action):
        beliefIndex = 4 * self.numberObjects
        attentionIndex = 4 * self.numberObjects + 6 * (self.numberObjects-1)

        oldPosition = oldState[:beliefIndex,]
        oldPosition = positionArraytToDataFrame(self.numberObjects, oldPosition)
        oldBelief = oldState[beliefIndex:attentionIndex,]
        oldBelief = beliefArrayToDataFrame(self.numberObjects, oldBelief, self.assumeWolfPrecisionList)
 
        wolfAction = self.wolfPolicy(oldPosition, self.wolfPrecision)
        distractorAction = self.distractorPolicy(oldPosition)
        actionForTransition = [action, wolfAction, distractorAction]
        
        newPosition = self.physicalTransition(oldPosition, actionForTransition)
        [newBelief, self.oldAttentionStatus] = self.updateBelief(
            oldBelief, oldPosition, newPosition, self.oldAttentionStatus, self.time + 1)

        newPositionArray = np.asarray(newPosition).flatten()
        newBeliefArray = np.asarray(newBelief).flatten()
        newState = np.concatenate((newPositionArray, newBeliefArray))

        self.time += 1

        return newState

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


if __name__ == '__main__':
	main()