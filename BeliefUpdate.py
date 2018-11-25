
import pandas as pd 
import numpy as np 
import scipy.stats as stats
import math
import datetime

def computeDeviationFromTrajectory(vector1,vector2):
	def calVectorNorm(vector):
		return np.power(np.power(vector, 2).sum(axis = 1), 0.5)
	innerProduct = np.dot(vector1, vector2.T).diagonal()
	angle = np.arccos(innerProduct/(calVectorNorm(vector1)*calVectorNorm(vector2)))
	return angle

def computeDeviationAngleDF(oldBelief,currentStates,sheepIdentity):
	oldStatesSelfDF=oldBelief.loc[sheepIdentity][['positionX','positionY','velocityX','velocityY']]
	oldStatesOthersDF=oldBelief.loc[1:][['positionX','positionY','velocityX','velocityY']]
	currentStatesSelfDF=currentStates.loc[sheepIdentity]
	currentStatesOthersDF=currentStates.loc[1:]
	assumeDirectionDF=(oldStatesSelfDF - oldStatesOthersDF).loc[:][['positionX','positionY']]
	observeDirectionDF=currentStatesOthersDF.loc[:][['velocityX','velocityY']]
	deviationAngle=computeDeviationFromTrajectory(assumeDirectionDF,observeDirectionDF)
	deviationAngleDF=pd.DataFrame(deviationAngle.values,index=assumeDirectionDF.index,columns=['chasingDeviation'])
	return deviationAngleDF

def createHypothesisInformationDF(assumeWolfPrecisionList,deviationAngleDF,oldBelief):
	objectIdentity=list(deviationAngleDF.index)
	hypothesisIndex=pd.MultiIndex.from_product([assumeWolfPrecisionList,objectIdentity],names=['assumeChasingPrecision','Identity'])
	hypothesisInformation=pd.DataFrame(list(deviationAngleDF.values)*len(assumeWolfPrecisionList),index=hypothesisIndex,columns=['chasingDeviation'])
	hypothesisInformation['pLogPrior']=list(oldBelief.loc[1:]['logP'].values)*len(assumeWolfPrecisionList)
	return hypothesisInformation

def initiateBeliefDF(initialStateList):
	beliefDF=pd.DataFrame(initialStateList,index=range(len(initialStateList)),columns=['positionX','positionY','velocityX','velocityY'])
	beliefDF['logP']=[0]+[1.0/(len(beliefDF.index)-1)]*(len(beliefDF.index)-1)
	return beliefDF

def computeLogLikelihood(deviationAngle,assumePrecision):
	pLogLikelihood = stats.vonmises.logpdf(deviationAngle, assumePrecision) + np.log(2) + np.log(math.pi)
	return pLogLikelihood

class BeliefUpdate():
	def __init__(self,assumeWolfPrecisionList,sheepIdentity):
		self.assumeWolfPrecisionList=assumeWolfPrecisionList
		self.sheepIdentity=sheepIdentity
	def __call__(self,oldBelief,currentStates):
		deviationAngleDF=computeDeviationAngleDF(oldBelief, currentStates, self.sheepIdentity)
		hypothesisInformation=createHypothesisInformationDF(self.assumeWolfPrecisionList, deviationAngleDF, oldBelief)
		hypothesisInformation['pLogLikelihood']=computeLogLikelihood(hypothesisInformation['chasingDeviation'].values,hypothesisInformation.index.get_level_values('assumeChasingPrecision'))
		hypothesisInformation['logP']=hypothesisInformation['pLogPrior'] + hypothesisInformation['pLogLikelihood']
		hypothesisInformation['logP']=np.log(np.exp(hypothesisInformation['logP'])/np.exp(hypothesisInformation['logP']).sum())
		posteriorList=[0]+list(np.log(np.exp(hypothesisInformation['logP']).groupby('Identity').sum().values))
		currentBelief=currentStates.copy()
		currentBelief['logP']=posteriorList
		return currentBelief

if __name__=="__main__":
	time0=datetime.datetime.now()
	import Transition
	statesList=[[10,10,0,0],[10,5,0,0],[15,15,0,0]]
	oldStates=pd.DataFrame(statesList,index=[0,1,2],columns=['positionX','positionY','velocityX','velocityY'])
	oldBelief=initiateBeliefDF(statesList)
	speedList=[5,3,3]
	currentActions=[[0,3],[0,3],[0,3]]
	movingRange=[0,0,15,15]
	assumeWolfPrecisionList=[50,1.3]
	sheepIdentity=0
	print('initialParameters',datetime.datetime.now()-time0)
	transState=Transition.Transition(movingRange, speedList)
	updateBelief=BeliefUpdate(assumeWolfPrecisionList, sheepIdentity)
	print('initialFunctions',datetime.datetime.now()-time0)
	currentStates=transState(oldStates, currentActions)
	currentBelief=updateBelief(oldBelief, currentStates)
	print('updateStatesAndBelief',datetime.datetime.now()-time0)
	print(oldBelief)
	print(currentStates)
	print(currentBelief)






