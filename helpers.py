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
sns.set_style("whitegrid")

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
    seriesData = removeOutliers(seriesData)
    n = len(seriesData)
    x = np.sort(seriesData)
    y = np.arange(1, n + 1) / n
    return x, y

## ----------------------------------------------------------------------------------------

def buildECDFs(df):
    ## This builds ECDFs for all numerical columns of the dataframe
    for col in df:        
        colData = df[col]
        colDataType = colData.dtypes

        if (colDataType == "float64"):
            x, y = getECDF(colData)
            _ = plt.plot(x, y, marker = ".", linestyle = "none")
            _ = plt.ylabel("ECDF")
            _ = plt.title("ECDF of " + col)
            _ = plt.margins(0.02)
            _ = plt.show()   
    return

## ----------------------------------------------------------------------------------------

def buildECDF(colData):
    # This builds a single ECDF for a series
    x, y = getECDF(colData)
    _ = plt.plot(x, y, marker = ".", linestyle = "none")
    _ = plt.ylabel("ECDF")
    _ = plt.margins(0.02)
    _ = plt.show()   
    return
            
## ----------------------------------------------------------------------------------------
            
def buildHistograms(df):
    ## This builds histograms for all numerical columns of the dataframe
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
            
def buildHistogram(colData):
    ## This builds histograms for all numerical columns of the dataframe
    colData = removeOutliers(colData)

    _ = plt.hist(colData, bins = 30)
    _ = plt.ylabel("Occurences")
    _ = plt.show()
    return
 
## ----------------------------------------------------------------------------------------

def getCorr(x, y):
    corr_mat = np.corrcoef(x, y)
    return corr_mat[0, 1]

## ----------------------------------------------------------------------------------------

def compareCDF(seriesA, seriesB):
    # Generate CDFs
    x_A, y_A = getECDF(seriesA)
    x_B, y_B = getECDF(seriesB)


    # plot CDFs
    _ = plt.plot(x_A, y_A, marker = ".", linestyle = "none", color = "blue")
    _ = plt.plot(x_B, y_B, marker = ".", linestyle = "none", color = "red")


    # Make 2% margin
    plt.margins(0.02)

    # Make a legend and show the plot
    _ = plt.legend(('SeriesA','SeriesB'), loc='lower right')
    _ = plt.ylabel("ECDF")
    plt.show()
    return

## ----------------------------------------------------------------------------------------

def bootstrap_replicate_1d(data, func):
    # generate bootstrap replicate of 1-dimensional data
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

## ----------------------------------------------------------------------------------------

def draw_bs_reps(data, func, size = 1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

## ----------------------------------------------------------------------------------------

# Permutation sampling is a great way to simulate the hypothesis that two variables have identical probability distributions.

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

## ----------------------------------------------------------------------------------------

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1) - np.mean(data_2)

    return diff

## ----------------------------------------------------------------------------------------

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

## ----------------------------------------------------------------------------------------

def identicalMeans(dataA, dataB, size):
    # draw permutation replicates
    x = size

    # compute the means of Winter and Summer earnings
    empirical_diff_means = diff_of_means(dataA, dataB)
    print("Empirical Difference in Means:", empirical_diff_means)

    perm_replicates = draw_perm_reps(dataA, dataB, diff_of_means, size = x)
    pvalue = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

    print("P-Value:", pvalue)
    return

