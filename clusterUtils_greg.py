import numpy as np
import utils
import scipy.spatial.distance
import sklearn.metrics as metrics
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import random
import glob
from datetime import datetime, timedelta


import pandas as pd
import numpy as np
import pickle
import re
import math
import glob
from datetime import datetime, timedelta



def convertToArray(C, numDives):
    """
    Parameters:
                C 
                    a dictionary that has a 
                    key: the cluster number
                        value: all the dive numbers that are in that cluster
                    numDives
                    an integer that says how many dives there are in this tag deployment.
    Returns: 
            labelVec
                    a vector that index i represents dive i's cluster that
                    it has been assigned to.
   """

    labelVec = np.zeros((numDives,))
    for key in C.keys():
        arrayOfDiveNumbersInThatCluster = C[key]
        for index in arrayOfDiveNumbersInThatCluster:
            labelVec[index] = key
    return labelVec



def GausMM(representations, maxNumClusters):
    numEval = 10
    minNumClusters = 2
    bic = np.zeros((numEval, maxNumClusters-minNumClusters+10))
    aic = np.zeros((numEval, maxNumClusters-minNumClusters+10))
    for j in range(0,numEval):
        for i in range(minNumClusters, maxNumClusters):
            gmm = GaussianMixture(n_components = i, covariance_type = 'full').fit(representations)
            tempBIC = gmm.bic(representations)
            tempAIC = gmm.aic(representations)
            bic[j,i-minNumClusters] = tempBIC
            aic[j,i-minNumClusters] = tempAIC
    return (aic, bic)
       



def kMedoids(D, k, tmax=1000):
    m,n = D.shape

    M = np.sort(np.random.choice(n,k))
    
    Mnew = np.copy(M)
    
    C = {}
    for t in range(tmax):
        #determine clusters, the arrays of data indices
        J = np.argmin(D[:,M],axis = 1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
            #update cluster medioids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa], C[kappa])], axis = 1)
            if J.shape != (0,):
                j = np.argmin(J)
                Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        

        #check for convergence
        if np.array_equal(M,Mnew):
            break
        M = np.copy(Mnew)
    else:
        #final update of cluster memberships
        J  = np.argmin(D[:M], axis = 1)
        for kappa in range(k):
            C[kappa] = np.where(J == kappa)[0]
    return M, C







def SC(representations, distanceMetric, maxNumClusters):
    '''
    Input: 
         representations: 
                         a matrix of size n x d, where n is the number of samples and d is the dimension of each sample
         distanceMetric:
                        a string representing the type of distance metric that should be used to compare samples when constructing the dissimilairity matrix.
         maxNumCluster: 
                         an integer saying what should be the upper bound on the interval between 2 and x of the number of clusters that is put into each call of kmediods. 
    
    '''
    if distanceMetric == "cosine": 
        distanceMat = scipy.spatial.distance.squareform(
            scipy.spatial.distance.pdist(representations, metric = 'cosine'))
    if distanceMetric == "euclidean": distanceMat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(representations, metric = 'euclidean'))
    if distanceMetric == "Jensen-Shannon": distanceMat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(representations, metric = utils.JSD))
    siloScores = []
    for i in range(2, maxNumClusters):
        M, C = kMedoids(distanceMat, k = i)
        labelsForEachDive = convertToArray(C, distanceMat.shape[0])
        siloScore = metrics.silhouette_score(distanceMat, labelsForEachDive, metric="precomputed", sampleSize = None) 
        siloScores += [siloScore]
    return siloScores
       
def reorder(labels, tag):
        '''
        input: 
              a list of length k (number of clusters) that has the dive numbers within them
                          [[1,54,5,7,36], [2,3,4], [23, 86, 45]]
        output: 
              a list of labels for each dive number
                          [1,2,2,2,2,3,1,3,3,3,1]
        '''
        labelsByDiveNum = np.zeros((1, len(tag.dives)))
        for i in range(0, len(labels)):
            divesInClust = labels[i]
            for j in range(0, len(divesInClust)):
                diveNum = divesInClust[j]
                labelsByDiveNum[0, diveNum] = i
        print(labelsByDiveNum)
        return labelsByDiveNum

def reorderInverse(labels):
    '''
    input: 
            labels [1,4,3,2,6,4,2,5,3,...,3,2,3]
    output: 
            a list of length k (number of clusters) that has the dive numbers within them
                          [[1,54,5,7,36], [2,3,4], [23, 86, 45]]
    '''
    def addToDict(myDict, key, value):
        if key in myDict.keys():
            tempList = myDict[key]
            tempList.append(value)
            myDict[key] = tempList
        else:
            tempList = list()
            tempList.append(value)
            myDict[key] = tempList
        return myDict
    
    labelToNumber = dict()
    for i in range(0, len(labels)):
        tempLabel = labels[i]
        addToDict(labelToNumber, tempLabel, i)
    listOfIndices = list()
    for key in sorted(labelToNumber.keys()):
        listOfIndices.append(labelToNumber[key])
    return listOfIndices
       
def collectExampleFromClusterAssignments(labels, tag):  
    
    '''
    HELPER FUNCTION
    '''
    
    def reorder(labels):
        '''
        input: 
              a list of length k (number of clusters) that has the dive numbers within them
                          [[1,54,5,7,36], [2,3,4], [23, 86, 45]]
        output: 
              a list of labels for each dive number
                          [1,2,2,2,2,3,1,3,3,3,1]
        '''
        labelsByDiveNum = np.zeros((1, len(tag.dives)))
        for i in range(0, len(labels)):
            divesInClust = labels[i]
            for j in range(0, len(divesInClust)):
                diveNum = divesInClust[j]
                labelsByDiveNum[diveNum] = i
        return labelsbyDiveNum
                          
  
    #labels =  reorder(labels)

    dives = tag.dives
    numLabels = len(labels)
    listOfExamples = [[] for i in range(0, numLabels)]
    depthChannelIndex = tag.findIndex("depth")
    for i in range(0, len(labels)):
        diveNumsInCurrCluster = labels[i]
        lst = listOfExamples[i]
        for j in range(0, len(diveNumsInCurrCluster)):
            diveNum =  diveNumsInCurrCluster[j]
            tempDive = dives[diveNum]
            depthsOfDive = tempDive.iloc[:, depthChannelIndex]
            lst.append(depthsOfDive)
        listOfExamples[i] = lst
    return listOfExamples


def plotExamples(listOfExamples, maxNumToPrint):
    idx = 0
    for examplesFromClusters in listOfExamples:
        numToPrint = min(maxNumToPrint, len(examplesFromClusters))
        f, tuplePlots = plt.subplots(1, numToPrint, sharey=True, figsize=(30,10), squeeze = False)
        for i in range(0, numToPrint):
            tuplePlots[0, i].plot(examplesFromClusters[i])
            plt.title('Examples from cluster number: ' + str(idx +1))
        tuplePlots[0, i].invert_yaxis()
        tuplePlots[0, i].set_title('Examples from cluster number: ' + str(idx +1))
        plt.show() 
        idx += 1
    
def simmMatrix(distanceMatrix, sigma):
    '''
    input: 
            distanceMatrix:
                            a n x n distance matrix
            sigma: 
                            hyperparameter to determine width of the graph
                            
    output: 
            a nxn simmilarity matrix
    
    '''
    simmMatrix = np.exp((-1. * distanceMatrix)/(2*sigma**2))
    return simmMatrix
               

def getDiveLocations(self,gis_data_path = "data/GIS/"):
        """
        Uses the lat/lon coordinates for each time stamp in fileName to assign (lat/lon) to each dive,
        specifically to the starting location of each dive.  If the GIS data does not exist for the given date
        it looks at neighboring dates both ahead and behind starting with 1 day ahead then 1 behind then 2 days ahead, etc.
        up to 20 days ahead (MAX_DAYS_AHEAD_TO_CHECK).
        Takes in an optional path to where the GIS folder is stored.
        returns and array of tuples [(lat1, lon1), (lat2, lon2) ...], where each tuple corresponds to the dives
        in self.dives.
        """
        MAX_DAYS_AHEAD_TO_CHECK = 20  #in case the dive day is not in the GIS data, look at subsequent days

        #searches for first 18 digits of tag number in folder
        possibleFiles = glob.glob(gis_data_path + self.name[0:18] + '*')
        assert len(possibleFiles) > 0, "No GIS data file found for tag: " + self.name
        fileName = possibleFiles[0]
        pdf = pd.read_csv(fileName, usecols = ['Longitude','Latitude','Date_'])
        pdf = pdf.set_index(['Date_'])

        timeCol = self.colNames.index('time')
        assert timeCol != -1, "Time column not found"
        locations = []
        for dive in self.dives:
            startTime = dive.iloc[0,timeCol]
            date = startTime.split()[0]
            found = False
            #check for given date with offset [0,+1,-1,+2,-2,...]. 0 is the date given in tag, +1 is next day.
            for day_ahead in [(-1)**i * (i//2) for i in range(1,MAX_DAYS_AHEAD_TO_CHECK*2)]:
                gis_date = (datetime.strptime(date, '%m/%d/%Y') + timedelta(days=day_ahead)).strftime('%-m/%-d/%Y')
                if gis_date in pdf.index:
                    lat = pdf.loc[gis_date,'Latitude']
                    lon = pdf.loc[gis_date,'Longitude']
                    locations.append((lat,lon))
                    found = True
                    break
            if (not found):
                locations.append((0,0))
                print("Unfound GIS data for tag",self.name,"and date",date)

        self.locations = locations
        return locations