#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm in ML, as shown in
examples/src/main/python/ml/kmeans_example.py.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys
import numpy as np
import scipy
import heapq
from scipy.sparse import csc_matrix
from scipy import sparse
from timeit import default_timer
import math
import numpy as np
from pyspark.sql import SparkSession


def parseVector(line):
    return np.array([float(x) for x in line.split(' ')])


def getCosineDistance(aArr , bArr):
    #num = sum(aArr * bArr)
    num = (aArr.multiply(bArr)).sum()
    den = math.sqrt((aArr.multiply(aArr)).sum()) * math.sqrt((bArr.multiply(bArr)).sum())
    return num / den

def closestPoint(p, centers):
    #print(p)
    #print(centers)
    bestIndex = 0

    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = -1*getCosineDistance(p,centers[i])
        if tempDist < closest:
            closest = tempDist
            bestIndex = i

    return bestIndex


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: kmeans <file> <k> <convergeDist> <outputfile>", file=sys.stderr)
        exit(-1)

    spark = SparkSession\
        .builder\
        .appName("PythonKMeans")\
        .getOrCreate()

    sc = spark.sparkContext;

    #inp = "data_s.txt"
    inp = sys.argv[1]
    inputFile = open(inp, "r")

    docCount = inputFile.readline()
    wordCount = inputFile.readline()
    numberWords = inputFile.readline()

    docCol = []
    wordCol = []
    countCol = []
    k = 3

    line = inputFile.readline()

    wordDict = {}

    while line:
        temp = []
        temp = line.split(" ")

        docCol.append(int(temp[0]) - 1)
        wordCol.append(int(temp[1]) - 1)
        countCol.append(int(temp[2]))

        if (int(temp[1]) - 1) not in wordDict:
            wordDict[(int(temp[1]) - 1)] = 1
        else:
            ct = wordDict[(int(temp[1]) - 1)]
            ct = ct + 1
            wordDict[(int(temp[1]) - 1)] = ct

        # print "DocId:"+temp[0]+" Word:"+temp[1]+" count:"+temp[2]
        line = inputFile.readline()

    row = np.array(docCol)
    col = np.array(wordCol)
    data = np.array(countCol)
    inpMatrix = csc_matrix((data, (row, col)), shape=(docCount, wordCount))

    idfDict = {}

    for key in wordDict:
        idfDict[key] = np.log2((float(docCount) + 1.0) / (wordDict[key] + 1.0))

    rowA = []
    wordCol = []
    wordIDF = []

    for key in idfDict:
        rowA.append(0)
        wordCol.append(key)
        wordIDF.append(idfDict[key])

    idfRow = csc_matrix((np.array(wordIDF), (np.array(rowA), np.array(wordCol))), shape=(1, wordCount))

    finalMatrix = inpMatrix.multiply(idfRow)

    tempMatrix = finalMatrix.copy()

    li = []
    for key in tempMatrix[0, :]:
        li.append(key)

    tempMatrix = tempMatrix.multiply(tempMatrix)

    normCol = []
    for j in xrange(0, int(docCount)):
        normCol.append(math.sqrt(tempMatrix[j, :].sum()))

    tempMatrix = finalMatrix.copy()
    for i in xrange(0, int(docCount)):
        tempMatrix[i, :] = tempMatrix[i, :] / normCol[i]

    #print(tempMatrix)
    #print(tempMatrix.toarray())
    
    inpL = []
    for i in xrange(0, int(docCount)):
        inpL.append(tempMatrix[i, :])
   
    #print(inpL)

    

    #lines = spark.read.text(sys.argv[1]).rdd.map(lambda r: r[0])
    
    data = sc.parallelize(inpL)
    K = int(sys.argv[2])
    convergeDist = float(sys.argv[3])

    kPoints = data.repartition(1).takeSample(False, K, 1)
    tempDist = 1.0

    while tempDist > convergeDist:
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p, 1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()

        #tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)
        tempDist = sum(math.sqrt(((kPoints[iK] - p).power(2)).sum()) for (iK, p) in newPoints)
        #print(tempDist)
        for (iK, p) in newPoints:
            kPoints[iK] = p

    #print("Final centers: " + str(kPoints))
    
    #FileWriting
    output = sys.argv[4]
    file = open(output,'w')
    for i in xrange(0,len(kPoints)):
	#print(kPoints[i].count_nonzero())
        file.write(str(kPoints[i].count_nonzero())+"\n")

    spark.stop()
