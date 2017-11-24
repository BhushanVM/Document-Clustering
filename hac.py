import numpy as np
import scipy
import heapq
from scipy.sparse import csc_matrix
from scipy import sparse
import math
import sys

inp = sys.argv[1]
k = int(sys.argv[2])

inputFile = open(inp,"r")

docCount = inputFile.readline()
wordCount = inputFile.readline()
numberWords = inputFile.readline()

docCol = []
wordCol = []
countCol = []

line = inputFile.readline()

wordDict = {}
docDict = {}

while line:
    temp = []
    temp = line.split(" ")

    docCol.append(int(temp[0])- 1)
    wordCol.append(int(temp[1]) - 1)
    countCol.append(int(temp[2]))

    if (tuple([(int(temp[0]) - 1)])) not in docDict:
        wl = []
        wl.append(int(temp[1])-1)
        docDict[tuple([(int(temp[0]) - 1)])] = wl
    else:
        wl = docDict[tuple([(int(temp[0]) - 1)])]
        wl.append(int(temp[1])-1)
        docDict[tuple([(int(temp[0]) - 1)])] = wl

    if (int(temp[1]) - 1) not in wordDict:
        wordDict[(int(temp[1]) - 1)] = 1
    else:
        ct = wordDict[(int(temp[1]) - 1)]
        ct = ct + 1
        wordDict[(int(temp[1]) - 1)] = ct

    #print "DocId:"+temp[0]+" Word:"+temp[1]+" count:"+temp[2]
    line = inputFile.readline()

row = np.array(docCol)
col = np.array(wordCol)
data = np.array(countCol)

inpMatrix = csc_matrix((data, (row, col)), shape=(docCount,wordCount))

idfDict = {}
for key in wordDict:
    idfDict[key] = np.log2((float(docCount)+1.0)/(wordDict[key]+1.0))

rowA = []
wordCol = []
wordIDF = []

for key in idfDict:
    rowA.append(0)
    wordCol.append(key)
    wordIDF.append(idfDict[key])

idfRow = csc_matrix((np.array(wordIDF), (np.array(rowA), np.array(wordCol))), shape=(1, wordCount))
#print idfRow

finalMatrix = inpMatrix.multiply(idfRow)
tempMatrix  = finalMatrix.copy()

li = []
for key in tempMatrix[0,:]:
    li.append(key)

tempMatrix = tempMatrix.multiply(tempMatrix)

normCol = []
for j in xrange(0,int(docCount)):
    normCol.append(math.sqrt(tempMatrix[j, :].sum()))

tempMatrix = finalMatrix.copy()
for i in xrange(0,int(docCount)):
    tempMatrix[i,:] = tempMatrix[i,:] / normCol[i]

nc = []
tMatrix = tempMatrix.multiply(tempMatrix)
for j in xrange(0,int(docCount)):
    nc.append(math.sqrt(tMatrix[j, :].sum()))

finalHeap = []
clusterDict = {}
tempDict = {}
denSqrSum = {}

for i in xrange(0,int(docCount)):
    l = [i]
    clusterDict[tuple(l)] = tempMatrix[i,:]
    tempDict[tuple(l)] = tempMatrix[i,:]
    denSqrSum[tuple(l)] = (tempMatrix[i,:].multiply(tempMatrix[i,:])).sum()

count = 0;
for i in xrange(0,int(docCount)):
    for j in xrange(i+1,int(docCount)):
        tempList = [tuple([i]), tuple([j])]
        s = set(docDict[tuple([i])]).intersection(set(docDict[tuple([j])]))
        sum = 0.0
        for val in s:
            sum += tempMatrix[i,val]*tempMatrix[j,val]

        heapq.heappush(finalHeap,(((sum/(nc[i]*nc[j]))*-1), tuple(tempList)))

heapq.heapify(finalHeap)

def getCentroidVector(tuple1,tuple2):
    div1 = len(tuple1)
    div2 = len(tuple2)
    return (tempDict[tuple1].multiply(div1) + tempDict[tuple2].multiply(div2))/(div1+div2)

def getCosineDistance(tuple1 , tuple2):
    s = set(docDict[tuple1]).intersection(set(docDict[tuple2]))
    num = 0.0
    for val in s:
        num += tempDict[tuple1][0,val] * tempDict[tuple2][0,val]

    den = math.sqrt(denSqrSum[tuple1]) * math.sqrt(denSqrSum[tuple2])
    return num / den

validClusters = set()
noClusters = len(clusterDict)
while(noClusters>k):
    popTuple = heapq.heappop(finalHeap)

    if not ((popTuple[1][0] in validClusters) or (popTuple[1][1] in validClusters)):
        del clusterDict[popTuple[1][0]]
        del clusterDict[popTuple[1][1]]

        #Compare with each other cluster
        newCluster = popTuple[1][0] + popTuple[1][1]
        tempLs = list(newCluster)

        tempCV = getCentroidVector(popTuple[1][0],popTuple[1][1])

        tempDict[newCluster] = tempCV
        denSqrSum[newCluster] = (tempCV.multiply(tempCV)).sum()

        docDict[newCluster] = list(set(docDict[popTuple[1][0]]).union(set(docDict[popTuple[1][1]])))

        for key in clusterDict:
            tempList = [newCluster,key]
            heapValidation = getCosineDistance(newCluster,key)
            heapq.heappush(finalHeap,(-1*heapValidation, tuple(tempList)))

        clusterDict[newCluster] = tempCV

        validClusters.add(popTuple[1][0])
        validClusters.add(popTuple[1][1])

        noClusters -= 1


for key in clusterDict:
    ans = map(lambda x: x + 1, list(key))
    print ",".join('%d'%x for x in ans)