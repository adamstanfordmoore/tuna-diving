import tagStruct
import numpy as np
from numpy.linalg import norm
import scipy.stats as ss
import scipy.spatial.distance
from sklearn.decomposition import PCA
import matplotlib.cm as mplcm
import matplotlib.colors as colors


datasets = [
        "5111027PAM110P0574TS.csv", 
        "5111033PAM110P0587-Archive.csv",
        "5111034PAM110P0588-Archive.csv", 
        "5111045PAM110P0590-Archive.csv", 
        "5112030PAM110P0416-Archive.csv", 
        "5112039PAM111P0762-Archive.csv", 
        "5112041PAM111P0763-Archive.csv"]

dataSetColNames = [
              
                    ["Time","Depth","Temperature","Light Level"], 
                    ["Time","Depth","Temperature","Light Level"],
                    ["Time","Depth","Temperature","Light Level"], 
                    ["Time","Depth","Temperature","Light Level"], 
                    ["Time","Depth","Temperature","Light Level"],
                    ["Time","Depth","Temperature","Light Level"], 
                    ["Time","Depth","Temperature","Light Level"]
                  ]




def JSD(P,Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (ss.entropy(_P, _M) + ss.entropy(_Q, _M))


def loadDatasets(threshold, truncationLength):
    '''
    HELPER FUNCTION
    '''
    def truncateDives(tags, minLength):
        for i in range(0, len(tags)):
            tag = tags[i]
            dives = tag.dives
            dives = [dives[k] for k in range(0, len(dives)) if len(dives[k]) > truncationLength]
            tag.dives = dives
        return tags
    '''
    HELPER FUNCTION
    '''
    
    tags = []
    for i in range(0, len(datasets)):
        path = "data/" + datasets[i]
        colNames = dataSetColNames[i]
        tag = tagStruct.Tag(path)
        tag.readcsv(colNames)
        tags.append(tag)
    dives = [tags[i].getDives(threshold) for i in range(0, len(tags))]
    tags = truncateDives(tags, truncationLength)
    return tags
    
    
def createSimpleRepresentations(tag):
    '''
    For each dive, extracts the following features: 
    1) average depth
    2) max depth
    3) percent of time at depths 0-1000 at 10m intervals
    '''
    listOfReps = list()
    depthChannelIndex = tag.findIndex("depth")
    velocityChannelIndex = tag.findIndex("velocity")
    for i in range(0, len(tag.dives)):
        diveDepths = tag.dives[i].iloc[:, depthChannelIndex]
        rowSum = np.sum(tag.diveCounts[i], axis = 1)/len(tag.dives[i])
        averageDepth = np.asarray([np.mean(diveDepths)])
        maxDepth = np.asarray([np.max(diveDepths)])
        averageVelo = np.asarray([np.mean(tag.dives[i].iloc[:, velocityChannelIndex])])
        print(diveDepths.shape)
        listOfReps.append(np.concatenate((rowSum, averageVelo, averageDepth, maxDepth)))
    tag.simpleReps = np.vstack(listOfReps)
    return np.vstack(listOfReps)
                          
    
    
    
def createRepresentation(tag, numComponents = 0, probability = False):
    '''
    HELPER FUNCTIONS
    '''
    def reduceCountsSVD(matrix, numComponents):
        pca = PCA()
        decomp = pca.fit(matrix)
        firstXLoadings = decomp.components_[:numComponents, :]
        reduced_x = matrix.dot(firstXLoadings.T)
        return reduced_x, decomp

    def matrixify(countReps):
        numSamples = len(countReps)
        columnNum = np.ndarray.flatten(countReps[0]).shape[0]
        matrixRep = np.zeros((numSamples, columnNum))
        for i in range(0, len(countReps)):
            matrixRep[i, :] = np.ndarray.flatten(countReps[i])
        return matrixRep
    '''
    HELPER FUNCTIONS
    '''
    
    tag.createCounts()
    if probability:
        matrix = matrixify(tag.diveCounts)
        matrix = matrix/matrix.sum(axis=1, keepdims=True)
        tag.ProbReps = matrix
        return matrix
    if numComponents == 0:
        return matrixify(tag.diveCounts)
    if numComponents != 0 and not probability: 
        matrix = matrixify(tag.diveCounts)
        tag.reps = matrix
        reducedMatrix, __decomp__ = reduceCountsSVD(matrix, numComponents)
        return reducedMatrix
    print("Error occurred")
    return None


def plotLocations(TAG,labels=None):
    """Plots dives with associated labels.  To avoid plotting unnecessary info
    this method only plots one dive of a unique (date,latitude,longitude,label)
    Labels go from 0 to 13"""

    MARKERS = ['o','^','>','p','+','s','d','x','<','P','*','D','H','o']
    NUM_COLORS = 12
    cm = plt.get_cmap('gist_rainbow')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    COLORS = [scalarMap.to_rgba(i) for i in range(NUM_COLORS)]
    MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

    if labels == None:
        labels = [0 for i in range(len(TAG.dives))]
    fig = plt.figure(figsize=(8, 8))
    m = Basemap(projection='lcc', resolution=None,
            width=4E6, height=4E6,
            lat_0=35, lon_0=-80,)
    m.etopo(scale=0.5, alpha=0.5)

    unique = set()

    for i,(lat,lon) in enumerate(TAG.locations):
        # Map (long, lat) to (x, y) for plotting
        if (TAG.diveDates[i],lat,lon,labels[i]) not in unique:
            unique.add((TAG.diveDates[i],lat,lon,labels[i]))
            x, y = m(lon, lat)
            month_index = int(TAG.diveDates[i].split('/')[0]) - 1
            label = MONTHS[month_index] + '/%s' % labels[i]
            plt.plot(x, y, marker=MARKERS[labels[i]], markersize=5,color = COLORS[month_index],label=label)



