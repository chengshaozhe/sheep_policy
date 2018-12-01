
import numpy as np 
import pandas as pd 

def modifyPrecisionForUntracked(attentionStatus,precisionPerSlot,precisionForUntracked):
	if attentionStatus==0:
		return precisionForUntracked/precisionPerSlot
	else:
		return attentionStatus

def modifyDecayForUntracked(attentionStatus,memoryratePerSlot,memoryrateForUntracked):
	if attentionStatus==0:
		return (1 - memoryratePerSlot)/((1 - memoryrateForUntracked)+0.00000001)
	else:
		return attentionStatus

class AttentionSwitch():
	def __init__(self,attentionLimitation):
		self.attentionLimitation=attentionLimitation
	def __call__(self,posteriorList,oldAttentionStatus):
		currentAttentionStatus=oldAttentionStatus.copy()
		currentAttentionStatusList=list(np.random.multinomial(self.attentionLimitation,posteriorList))*oldAttentionStatus.groupby(['Identity']).size().values[0]
		currentAttentionStatus['attentionStatus']=np.array(currentAttentionStatusList)
		return currentAttentionStatus

class AttentionToPrecisionAndDecay():
	def __init__(self,precisionPerSlot,precisionForUntracked,memoryratePerSlot,memoryrateForUntracked):
		self.precisionPerSlot = precisionPerSlot
		self.precisionForUntracked = precisionForUntracked
		self.memoryratePerSlot = memoryratePerSlot
		self.memoryrateForUntracked = memoryrateForUntracked
	def __call__(self,attentionStatus):
		attentionForPrecision = list(map(lambda x: modifyPrecisionForUntracked(x,self.precisionPerSlot,self.precisionForUntracked),attentionStatus['attentionStatus'].values))
		attentionForDecay = list(map(lambda x: modifyDecayForUntracked(x,self.memoryratePerSlot,self.memoryrateForUntracked),attentionStatus['attentionStatus'].values))
		precisionHypothesis = np.multiply(self.precisionPerSlot , attentionForPrecision)+0.00000001
		decayHypothesis = 1 - np.divide((1 - self.memoryratePerSlot),np.add(attentionForDecay,0.00000001))
		# decayHypothesis = 1 - np.divide((1 - self.memoryratePerSlot),(np.power(2,np.array(attentionForDecay)-1))+0.00000001)
		precisionHypothesisDF = pd.DataFrame(precisionHypothesis,index=attentionStatus.index,columns=['perceptionPrecision'])
		decayHypothesisDF = pd.DataFrame(decayHypothesis,index=attentionStatus.index,columns=['memoryDecay'])
		return precisionHypothesisDF, decayHypothesisDF
