
import pandas as pd 
import numpy as np 

def renormalVector(rawVector,targetLength):
	rawLength=np.power(np.power(rawVector, 2).sum(), 0.5)
	changeRate=np.divide(targetLength,rawLength)
	return np.multiply(rawVector,changeRate)

class WolfPolicy():
	def __init__(self,sheepIdentity,wolfIdentity,speed):
		self.sheepIdentity=sheepIdentity
		self.wolfIdentity=wolfIdentity
		self.speed=speed
	def __call__(self,states,wolfPrecision):
		wolfState=states.xs(self.wolfIdentity).values
		sheepState=states.xs(self.sheepIdentity).values
		wolfPosition = wolfState[0:2]
		sheepPosition = sheepState[0:2]
		direction = np.subtract(sheepPosition, wolfPosition)
		theta = np.random.vonmises(0,wolfPrecision)
		action=[np.multiply(direction,[np.cos(theta),-np.sin(theta)]).sum(),np.multiply(direction,[np.sin(theta),np.cos(theta)]).sum()]
		action=renormalVector(action, self.speed)
		return action

class DistractorPolicy():
	def __init__(self,distractorIdentity,distractorPrecision,speed):
		self.distractorIdentity=distractorIdentity
		self.distractorPrecision=distractorPrecision
		self.speed=speed
	def __call__(self,states):
		distractorState=states.xs(self.distractorIdentity).values
		distractorVelocity=distractorState[2:]
		if distractorVelocity[0]==0 & distractorVelocity[1]==0:
			distractorVelocity=np.array([np.random.uniform(-1,1),np.random.uniform(-1,1)])
		theta = np.random.vonmises(0,self.distractorPrecision)
		action=[np.multiply(distractorVelocity,[np.cos(theta),-np.sin(theta)]).sum(),np.multiply(distractorVelocity,[np.sin(theta),np.cos(theta)]).sum()]
		action=renormalVector(action, self.speed)
		return action

if __name__=='__main__':
	statesList=[[10,10,0,0],[10,5,0,0],[15,15,0,0]]
	statesDF=pd.DataFrame(statesList,index=[0,1,2],columns=['positionX','positionY','velocityX','velocityY'])
	sheepIdentity=1
	wolfIdentity=0
	distractorIdentity=2
	wolfPrecision=50
	distractorPrecision=0.5/3.14
	speed=3

	takeWolfAction=WolfPolicy(sheepIdentity, wolfIdentity, speed)
	takeDistractorAction=DistractorPolicy(distractorIdentity, distractorPrecision, speed)

	wolfAction=takeWolfAction(statesDF, wolfPrecision)
	print(wolfAction)
	distractorAction=takeDistractorAction(statesDF)
	print(distractorAction)
	pass






