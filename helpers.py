# helper functions

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from scipy import stats
import time

# set seaborn settings
sns.set()
plt.rcParams["patch.force_edgecolor"] = True # set lines

import warnings
warnings.filterwarnings('ignore')

def test():
    print(True)
    return

def convertDataType(df):    
    df["taxi_id"] = df["taxi_id"].astype("category")
    df["tolls"] = df["tolls"].astype("float").fillna(0.0)
    df["trip_start_timestamp"] = pd.to_datetime(df["trip_start_timestamp"], errors = "coerce").dt.strftime("%m/%d/%Y %I:%M %p")
    df["trip_end_timestamp"] = pd.to_datetime(df["trip_end_timestamp"], errors = "coerce").dt.strftime("%m/%d/%Y %I:%M %p")
    df["company"] = df["company"].astype("category")
    df["dropoff_centroid_longitude"] = df["dropoff_centroid_longitude"].astype("category")
    df["dropoff_centroid_latitude"] = df["dropoff_centroid_latitude"].astype("category")
    df["pickup_centroid_latitude"] = df["pickup_centroid_latitude"].astype("category")
    df["pickup_centroid_longitude"] = df["pickup_centroid_longitude"].astype("category")
    df["fare"] = df["fare"].astype("float").fillna(0.0)
    df["payment_type"] = df["payment_type"].astype("category")
    df["dropoff_community_area"] = df["dropoff_community_area"].astype("category")
    df["pickup_community_area"] = df["pickup_community_area"].astype("category")
    df["tips"] = df["tips"].astype("float").fillna(0.0)
    df["trip_miles"] = df["trip_miles"].astype("float").fillna(0.0)
    df["trip_seconds"] = df["trip_seconds"].astype("float").fillna(0.0)
    df["trip_total"] = df["trip_total"].astype("float").fillna(0.0)   
    return(df)

## ----------------------------------------------------------------------------------------

def removeOutliers(seriesData):   
    seriesData = seriesData.astype(float).fillna(0.0)
    Q75, Q25 = np.percentile(seriesData, [75, 25])
    IQR = Q75 - Q25
    min = Q25 - (IQR * 1.5)
    max = Q75 + (IQR * 1.5)
    seriesData = np.array(seriesData)
    result = seriesData[np.where((seriesData >= min) & (seriesData <= max))]
    return(result)

## ----------------------------------------------------------------------------------------

def getECDF(seriesData):
    n = len(seriesData)
    x = np.sort(seriesData)
    y = np.arange(1, n + 1) / n
    return x, y

## ----------------------------------------------------------------------------------------

def buildECDF(df):
    for col in df:        
        colData = df[col]
        colDataType = colData.dtypes

        if (colDataType == "float64"):
            colData = removeOutliers(colData)
            x, y = getECDF(colData)
            _ = plt.plot(x, y, marker = ".", linestyle = "none")
            _ = plt.xlabel("Percent of " + col)
            _ = plt.ylabel("ECDF")
            _ = plt.title("ECDF of " + col)
            _ = plt.margins(0.02)
            _ = plt.show()   
    return
            
## ----------------------------------------------------------------------------------------
            
def buildHistograms(df):
    for col in df:        
        colData = df[col]
        colDataType = colData.dtypes
        
        if (colDataType == "float64"):
            colData = removeOutliers(colData)
            
            _ = plt.hist(colData, bins = 30)
            _ = plt.xlabel(col)
            _ = plt.ylabel("Occurences")
            _ = plt.title("Histogram of " + col)
            _ = plt.show()
    return
 
## ----------------------------------------------------------------------------------------

def getCorr(x, y):
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0, 1]

## ----------------------------------------------------------------------------------------