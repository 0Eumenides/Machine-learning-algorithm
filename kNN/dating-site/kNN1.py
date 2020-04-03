import numpy as np
import operator
def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels
def classify0(inX,dataSet,labels,k): #inX是待判断数据
    dataSetSize=dataSet.shape[0]
    diffMat=np.tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDisttances=sqDiffMat.sum(axis=1)
    distances=sqDisttances**0.5
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        votelabel=labels[sortedDistIndicies[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
