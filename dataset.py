import pandas as pd
import numpy as np


# This function reads dataset.
def readingDataset(fileName, stockType, Features):
    
    fileName='./data/'+fileName
    
    if stockType=='IranStock':
        dataset=pd.read_csv(fileName, usecols=Features, engine='python')
        dataset = dataset.reindex(index = dataset.index[::-1])
        dataset=dataset.dropna(how='any')
        print('Shape of dataset: ', dataset.shape)
    
    if stockType=='Forex':
        dataset=pd.read_csv(fileName, usecols=Features, engine='python')
        dataset=dataset.dropna(how='any')
        print('Shape of dataset: ', dataset.shape)       
        
    return dataset


def normalize(df, target):
    result = df.copy()
    
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        
    max_target = df[target].max()
    min_target = df[target].min()
        
    return result, max_target, min_target


def true_scale(df, _max, _min):    
    true_value = []
    for i in range(len(df)):
        true_value.append(df[i]*(_max-_min)+_min )
    true_value = np.array(true_value)
    return true_value


def trainTestSplit(dataset):
    
    train_size = int(len(dataset) * 0.7)
    #test_size = len(dataset) - train_size
    
    datasetTrain = dataset[0:train_size]
    print('Train dataset has shape of: ', datasetTrain.shape)
    
    datasetTest = dataset[train_size:]
    print('Train dataset has shape of: ', datasetTest.shape)
    
    return np.array(datasetTrain), np.array(datasetTest)

def generateDataset (data, data_target, look_back, forward_days):
    
    x=[]
    y=[]
    
    for i in range(0,len(data) -look_back -forward_days +1, 1):
        x.append(data[i:(i+look_back)])
        y.append(data_target[(i+look_back):(i+look_back+forward_days)])
        
    x, y = np.array(x), np.array(y)
    
    #y = np.array([list(a.ravel()) for a in y])
    
    print('shape of xtrain: ', x.shape, ' and shape of ytrain: ',  y.shape)
        
    return x, y



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    