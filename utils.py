import numpy as np
import scipy.stats as ss
from numpy.linalg import norm
from mpl_toolkits.basemap import Basemap  # may have to execute $ conda install basemap
import matplotlib.cm as mplcm
import matplotlib.colors as colors

def JSD(P,Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (ss.entropy(_P, _M) + ss.entropy(_Q, _M))

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

