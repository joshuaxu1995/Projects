import arff
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import operator
from collections import defaultdict
from statsmodels.compat import counter

conf_thres = 0.4
startIndice = 1000
numPoints = 400
sampleSize = 500
dataDict = defaultdict(list)
means = dict()

for row in arff.load('EEG Eye State.arff'):
    for col in range(len(row)-1):
        dataDict[col].append(row[col])

for key in dataDict.keys():
    dataDict[key] = dataDict[key][startIndice: startIndice + numPoints];
    means[key] = np.mean(dataDict[key])


def findCumSum(myList):
        methodAverage = np.mean(myList)
        cumulSum = [0]
        for (key,value) in enumerate(myList):
            cumulSum.append(cumulSum[key-1]+ value - methodAverage)
            min_index, min_value = min(enumerate(cumulSum), key=operator.itemgetter(1))
            max_index, max_value = max(enumerate(cumulSum), key=operator.itemgetter(1))
            Diff = max_value - min_value   
            #fig = plt.figure() 
            #plotGraph(cumulSum, fig)
        return [Diff, max_index]

def plotGraph(myList, fig):
#     fig = plt.figure()
        plt.plot(myList)  
        plt.ylabel('CumulSum Value', figure = fig)
        plt.xlabel('Sample number', figure = fig)
        plt.show()
        return


def plotGraphwithChangePoints(myList, fig, changePoints):
#     fig = plt.figure()
        plt.plot(myList)
        print(myList[38])
        for (i,c,_) in changePoints:
            plt.plot(i, myList[i], figure = fig,linestyle='--', marker='o', color=(1-c,1-c,0))
#             print(i)
#             print(myList[i])    
        plt.ylabel('CumulSum Value', figure = fig)
        plt.xlabel('Sample number', figure = fig)
        plt.show()
        return

""" BOOTSTRAP WITHOUT REPLACEMENT - BUT ONLY FOR DICTIONARIES- NEXT METHOD IS FOR LISTS"""


def notABootStrapDicts(myDict):

    arrayProbs = []
    for key in myDict.keys():
        origDiff, changePoint = findCumSum(myDict[key])
        num = 0
        for count in range(sampleSize):
            newList = myDict[key]
            random.shuffle(newList)
            newDiff, indexChange = findCumSum(myDict[key])
            if newDiff < origDiff:
                num = num+1
        num = num/sampleSize
       #print (num)
    arrayProbs.append(num)
    return arrayProbs


""" BOOTSTRAP WITHOUT REPLACEMENT - BUT ONLY FOR LISTS"""

def notABootStrapList(myList):
    
    origDiff, changePoint = findCumSum(myList)
    num = 0
    for count in range(sampleSize):
        newList = myList[:]
        random.shuffle(newList)
        #print("Orig: {0}, new: {1}".format(myList,newList))
        newDiff, indexChange = findCumSum(newList)
        if newDiff < origDiff:
            num = num+1
    num = num/sampleSize
    #print (num)
    return num

def MSE(myList):
    x_bar = np.mean(myList)
    MSE = 0.0
    for entry,value in enumerate(myList):
        MSE = MSE + np.power((x_bar - value),2)
    #MSE = MSE/ float(len(myList))
    #print("returning MSE ", MSE)
    return MSE


 

def minimizeMSE(d,myList,shift,prevConfidence):
    allMSE = []
    #print("At level {0} ({1} left) running results: {2}".format(d,len(myList),shift))
    confLevel = notABootStrapList(myList) * prevConfidence
    #print("Confidence level of change of {0}.".format(confLevel))
    if len(myList) == 3:
        if confLevel > conf_thres:
            return [((1+shift,confLevel,d))]
        else:
            return []
    elif len(myList) < 3:
        return []
    else:
        for i in range(1,len(myList)-2):
            MSE1 = MSE(myList[0:i+1])
            MSE2 = MSE(myList[i+1:len(myList)])
            MSETotal = MSE1 + MSE2
            #print("Returning MSTotal Index {0} found with score {1}".format(i,MSETotal))
            allMSE.append((MSETotal,i+1))
        #print("newly found: ", allMSE)
        (value, index) = min(allMSE)
        #print("Layer{0}, Best index {1} found with score {2}, with shift {3}".format(d,index,value,shift))
        #res = [((index+shift,value))]
        if confLevel > conf_thres:
            res = [(index+shift,confLevel,d)]
        else:
            res = []
        res.extend(minimizeMSE(d+1,myList[0:index+1],shift,confLevel))
        res.extend(minimizeMSE(d+1,myList[index+1:len(myList)], index+shift,confLevel))
        return res
        #allCP.append((index+shift,value))
        #return minimizeMSE(d+1,myList[0:index+1], allCP,0) + minimizeMSE(d+1,myList[index+1:len(myList)], [], index)    



# cumulSum = [0]
# for key,value in enumerate(mydict):
#     cumulSum.append(cumulSum[key-1]+ mydict[key] - d)

#print (cumulSum)    

# fig5 = plt.figure(5)
# count = 0
# for key in dataDict.keys():  
#     plotGraph(dataDict[key], plt.figure(count))
#     count+=1
    
    

#print(MSE(dataDict))
i= 0
# for key in dataDict.keys():
i = i+1
res_i =0
changePoints= minimizeMSE(0,dataDict[0],0,1)
change_point_dict = {d:[(i_,c_,d_) for (i_,c_,d_) in changePoints if d_==d] for (i,c,d) in changePoints}
for (d,d_list) in change_point_dict.items():
    print("At level {0}: {1}".format(d,sorted(d_list,key=operator.itemgetter(0))))
#for res in changePoints.sort(key=lambda tup: tup[2]):
    
changePoint_indices = [int(ci[0]) for ci in changePoints]
#print(changePoints)    
#print(dataDict[0])
# plotGraph(dataDict[0],plt.figure(i))
plotGraphwithChangePoints(dataDict[0],plt.figure(i),changePoints)
plt.plot()
    
