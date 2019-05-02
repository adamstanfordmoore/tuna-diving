import pandas as pd
import numpy as np
import pickle
import re
import math

class Tag(object):
    """A Tag is an object with properties that we will probably want when dealing with analysis. 

    Attributes:
        name: A string representing tag's name.
        numDays: An integer representing the number of days in the tag's deployment
        __raw__: The csv format in its rawest form
    """



    def __init__(self, fileString):
        """
        Either returns the last-saved instance of the tag or creates a new instance.
        """
        self.filePath = fileString
        self.name = self.getName()
        self.load()
       # self.save()

        

    def save(self):
        '''
        Saves the instance of this tag
        '''
        f = open("class instances/" + self.name + ".txt","wb+")
        pickle.dump(self,f) 
        f.close()

    def getDives(self, threshold):
        '''
        Returns list of nonoverlapping dives in the tag deployment that correspond to dives.
        Also sets self.dives to be this list as well.  
        Input: 
             Threshold: Depth at which one should start and end dives with. 
        Potential hyperparameter: How long should the dive be? Truncate samples based on this. 
        
        '''

        depthColumnIndex = self.findIndex('depth')
        depthChannel = self.colNames[depthColumnIndex]
        onDive = False
        startTime = "fjdkslfd"
        endTime = "fdjsklfd"
        listOfDives = list()
        for row in self.__raw__.itertuples(index=True, name='Pandas'):
            currDepth = getattr(row, depthChannel)
            if not(onDive and currDepth >= threshold):
                if (not(onDive)) and currDepth >= threshold:
                    onDive = True 
                    startTime =  row[0]
                elif onDive and currDepth < threshold:
                    onDive = False
                    endTime = row[0]
                    wholeDive = self.__raw__.iloc[startTime:endTime, ] #This will be a pandas dataframe...beware of numpy-like indexing
                    listOfDives.append(wholeDive)
        self.dives = listOfDives
        self.threshold = threshold
        self.save()
        return listOfDives
    
    def getDiveLocations(self,fileName):
        """
        Uses the lat/lon coordinates for each time stamp in fileName to assign (lat/lon) to each dive,
        specifically to the starting location of each dive.
        returns and array of tuples [(lat1, lon1), (lat2, lon2) ...], where each tuple corresponds to the dives
        in self.dives.
        """
        locations = []
        for dive in self.dives:
            pass


        self.locations = locations
        return locations

    def addVelocityColumn(self):
        dives = self.dives
        depthChannelIndex = self.findIndex("depth")
        for i in range(0, len(dives)):
            dive = dives[i]
            depthChannel = dive[self.colNames[depthChannelIndex]].values
            velocity = np.asarray([0] + [depthChannel[i] - depthChannel[i-1] for i in range(1, len(depthChannel))])
            dive["velocity"] = pd.Series(velocity , index=dive.index)
            dives[i] = dive
        print("finished")
        self.colNames = list(self.colNames) + ["velocity"]
        self.dives = dives
    
    def find_nearest(self, array,value):
        '''
        Performs fast binary search on array with value in-hand. 
        returns value in array that is closest to value. 
        assumes that array is sorted (which it will be in this implementation)
        '''
        
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

              

          
    def createCounts(self):
        depthChannelIndex = self.findIndex("depth")
        velocityChannelIndex = self.findIndex("velocity")
        dives = self.dives
        depthValues = np.linspace(self.threshold, 1200, 100)
        velocityValues = np.linspace(-20, 20, 100)
        listOfCounts = []
        for dive in dives:
            counts = np.zeros((100,100))
            for j in range(0, len(dive)):
                idx1 = self.find_nearest(depthValues, dive.iloc[j, depthChannelIndex])
                idx2 = self.find_nearest(velocityValues, dive.iloc[j, velocityChannelIndex])
                counts[idx1, idx2] += 1
            listOfCounts.append(counts)
        self.diveCounts = listOfCounts
        return listOfCounts
                                        
                          
            
            
    


    def findIndex(self, string):
        '''
        Returns the index of the column that best matches the string that is passed in. 
        Returns first column that has the request substring. If none of them have it, then return -1. 
        '''

        for i in range(0, len(self.colNames)):
            if self.colNames[i].find(string) != -1:
                return i

        return -1

   


    def load(self):
        '''
        Loads the instance of the tag
        '''
        try:
            f = open("class instances/" + self.name +'.txt','r')
            self  = pickle.load(f)
            return 0
        except:
            return -1

        
     

    
    def readcsv(self, colNames):
        '''
        Reads the csv file and stores it as a matrix. 

        colNames - 
                    Should be an array of strings that denote the names of the columns that you want to use. 
                    Note that the column names must exactly match (with correct cases) the columns names in the .csv that you are loading.
                    Example: colNames = ['date', 'time', 'depth (m)', 'internal temp', 'light']

        Returns the matrix
        '''
        pdf = pd.read_csv(self.filePath, usecols = colNames)
        toLower = [colNames[i].lower() for i in range(0, len(colNames))] 
        replaceDict = {colNames[j] : toLower[j] for j in range(0, len(colNames))}
        pdf.rename(columns = replaceDict)
        pdf.columns = toLower
        #you are going to have to be handling some fork util function here on how to get the right time format. 
        self.__raw__ = pdf
        self.colNames = toLower 
        print(self.colNames)
        print("found: ", self.colNames[self.findIndex("time")])
        self.__raw__.set_index(self.colNames[self.findIndex('time')])
        self.save()
         




    def getName(self):
        '''
        Input: filePath
                        A string that is the path that is local within the computer for the tag that you want to use. 
        Output: name
                        The string that represents the tag name. 
        This works by taking the substring of the filepath of the last occurence of the '/' 
        character and chopping off the last 4 characters that represent the '.csv' part of the filepath. 
        '''
        print(self.filePath)
        indices = [m.start() for m in re.finditer('/', self.filePath)]
        if(len(indices) == 0):
            name = self.filePath[:len(filePath) - 4]
            return name
        else:
            name = self.filePath[indices[len(indices)-1] +1: len(self.filePath) - 4]
            return name





