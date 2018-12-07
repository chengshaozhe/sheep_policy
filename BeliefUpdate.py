
import pandas as pd 
import numpy as np 
import scipy.stats as stats
import math
import Attention

def computeDeviationFromTrajectory(vector1,vector2):
	def calVectorNorm(vector):
		return np.power(np.power(vector, 2).sum(axis = 1), 0.5)
	innerProduct = np.dot(vector1, vector2.T).diagonal()
	angle = np.arccos(innerProduct/(calVectorNorm(vector1)*calVectorNorm(vector2)))
	return angle

def initiateBeliefDF(objectsNumber,assumeWolfPrecisionList):
	multiIndex=pd.MultiIndex.from_product([assumeWolfPrecisionList,range(1,objectsNumber)],names=['assumeChasingPrecision','Identity'])
	initialProbability=[1.0/len(multiIndex)]*len(multiIndex)
	beliefDF=pd.DataFrame(initialProbability,index=multiIndex,columns=['p'])
	return beliefDF

def initiateAttentionStatus(oldBelief,attentionLimitation):
	attentionStatusList=list(np.random.multinomial(attentionLimitation,oldBelief.groupby(['Identity']).sum()['p'].values))*len(oldBelief.groupby('assumeChasingPrecision').sum().index)
	objectIdentity=list(oldBelief.groupby('Identity').sum().index)
	assumeWolfPrecisionList=list(oldBelief.groupby('assumeChasingPrecision').sum().index)
	attentionStatus=pd.DataFrame(attentionStatusList,index=oldBelief.index,columns=['attentionStatus'])
	return attentionStatus

def computeDeviationAngleDF(oldStates,currentStates,sheepIdentity):
	oldStatesSelfDF=oldStates.loc[sheepIdentity][['positionX','positionY','velocityX','velocityY']]
	oldStatesOthersDF=oldStates.loc[1:][['positionX','positionY','velocityX','velocityY']]
	currentStatesSelfDF=currentStates.loc[sheepIdentity]
	currentStatesOthersDF=currentStates.loc[1:]
	assumeDirectionDF=(oldStatesSelfDF - oldStatesOthersDF).loc[:][['positionX','positionY']]
	observeDirectionDF=currentStatesOthersDF.loc[:][['velocityX','velocityY']]
	deviationAngle=computeDeviationFromTrajectory(assumeDirectionDF,observeDirectionDF)
	deviationAngleDF=pd.DataFrame(deviationAngle.values,index=assumeDirectionDF.index,columns=['chasingDeviation'])
	return deviationAngleDF

def createHypothesisInformationDF(deviationAngleDF,oldBelief,precisionStatusDF,decayStatusDF):
	objectIdentity=list(deviationAngleDF.index)
	assumeWolfPrecisionList=list(oldBelief.groupby('assumeChasingPrecision').sum().index)
	hypothesisIndex=oldBelief.index
	hypothesisInformation=pd.DataFrame(list(deviationAngleDF.values)*len(assumeWolfPrecisionList),index=hypothesisIndex,columns=['chasingDeviation'])
	hypothesisInformation['pPrior']=list(oldBelief['p'].values)
	hypothesisInformation['perceptionPrecision'] = precisionStatusDF['perceptionPrecision'].values
	hypothesisInformation['memoryDecay'] = decayStatusDF['memoryDecay'].values
	return hypothesisInformation

def computeLikelihood(deviationAngle,assumePrecision):
	pLikelihood = stats.vonmises.pdf(deviationAngle, assumePrecision)*2*math.pi
	return pLikelihood

class BeliefUpdateWithAttention():
	def __init__(self,computePrecisionAndDecay,switchAttention,attentionSwitchFrequency,sheepIdentity):
		self.computePrecisionAndDecay=computePrecisionAndDecay
		self.switchAttention=switchAttention
		self.attentionSwitchFrequency=attentionSwitchFrequency
		self.sheepIdentity=sheepIdentity
		pass
	def __call__(self,oldBelief,oldStates,currentStates,oldAttentionStatus,currentTime):
		deviationAngleDF=computeDeviationAngleDF(oldStates, currentStates, self.sheepIdentity)
		[precisionStatusDF,decayStatusDF]=self.computePrecisionAndDecay(oldAttentionStatus)
		hypothesisInformation=createHypothesisInformationDF(deviationAngleDF, oldBelief, precisionStatusDF, decayStatusDF)
		hypothesisInformation['pLikelihood']=computeLikelihood(hypothesisInformation['chasingDeviation'].values,1/(1/hypothesisInformation.index.get_level_values('assumeChasingPrecision') + 1/hypothesisInformation['perceptionPrecision']))
		hypothesisInformation['p']=np.power(hypothesisInformation['pPrior'],hypothesisInformation['memoryDecay']) * hypothesisInformation['pLikelihood']
		hypothesisInformation['p']=hypothesisInformation['p']/hypothesisInformation['p'].sum()
		currentBelief=oldBelief.copy()
		currentBelief['p']=hypothesisInformation['p']
		if np.mod(currentTime,self.attentionSwitchFrequency)==0:
			currentAttentionStatus=self.switchAttention(posteriorList,oldAttentionStatus)
		else:
			currentAttentionStatus=oldAttentionStatus.copy()
		return currentBelief,currentAttentionStatus

if __name__=="__main__":
	import datetime
	time0=datetime.datetime.now()
	import Transition
	objectsNumber=3
	statesList=[[10,10,0,0],[10,5,0,0],[15,15,0,0]]
	oldStates=pd.DataFrame(statesList,index=[0,1,2],columns=['positionX','positionY','velocityX','velocityY'])
	speedList=[5,3,3]
	currentActions=[[0,3],[0,3],[0,3]]
	movingRange=[0,0,15,15]
	assumeWolfPrecisionList=[50,1.3]
	attentionLimitation=2
	sheepIdentity=0
	oldBelief=initiateBeliefDF(objectsNumber, assumeWolfPrecisionList)
	oldAttentionStatus=initiateAttentionStatus(oldBelief, attentionLimitation)
	precisionPerSlot=8.0
	precisionForUntracked=2.5
	memoryratePerSlot=0.7
	memoryrateForUntracked=0.45
	attentionSwitchFrequency=12
	print('initialParameters',datetime.datetime.now()-time0)
	transState=Transition.Transition(movingRange, speedList)
	computePrecisionAndDecay=Attention.AttentionToPrecisionAndDecay(precisionPerSlot, precisionForUntracked, memoryratePerSlot, memoryrateForUntracked)
	switchAttention=Attention.AttentionSwitch(attentionLimitation)
	updateBelief=BeliefUpdateWithAttention(computePrecisionAndDecay, switchAttention, attentionSwitchFrequency, sheepIdentity)
	print('initialFunctions',datetime.datetime.now()-time0)
	currentStates=transState(oldStates, currentActions)
	print('updateState',datetime.datetime.now()-time0)
	[currentBelief,currentAttentionStatus]=updateBelief(oldBelief,oldStates,currentStates,oldAttentionStatus,1)
	print('updateBelief',datetime.datetime.now()-time0)
	print(oldBelief)
	print(currentStates)
	print(currentBelief)



