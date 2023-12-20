from fastdist import fastdist
from numpy import trapz
import numpy as np


def average_distance(rdf ,rmesh, rrange):
    x,y = np.asarray([l for l in np.asarray([rmesh, rdf]).transpose() if rrange[0]<l[0]<rrange[1]]).transpose()
    return np.sum(x*y)/np.sum(y)

def make_rdf_feff(distances, rmeshPrime):
    digitized =np.digitize(distances
        , rmeshPrime)
    unique, counts = np.unique(digitized, return_counts=True)
    counter = [0]*len(rmeshPrime)
    for i in range(len(unique)):
        counter[unique[i]-1] = counts[i]/(rmeshPrime[1]-rmeshPrime[0])
    return np.asarray(counter)

def integrate_mono(rdf ,rmesh, rrange):
    x,y = np.asarray([l for l in np.asarray([rmesh, rdf]).transpose() if rrange[0]<l[0]<rrange[1]]).transpose()
    return trapz(y,x)