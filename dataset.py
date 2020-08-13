import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler



# This function reads dataset.
def readingDataset(fileName, stockType):
    
    if stockType=='IranStock':
        dataset=pd.read_csv(fileName, usecols=[2,3,4,5,7], engine='python')
        dataset = dataset.reindex(index = dataset.index[::-1])
        print('Shape of dataset: ', dataset.shape)
    
    if stockType=='Forex':
        dataset=pd.read_csv(fileName, usecols=[1,2,3,4,6], engine='python')
        print('Shape of dataset: ', dataset.shape)       
        
    return dataset


def closePrice (fileName, stockType):
    
    dataset=readingDataset(fileName, stockType)
    dataset=dataset['Close']
    dataset=np.array(dataset)
    dataset=np.reshape(dataset,(len(dataset),1))
    print('Dataset has shape of: ', dataset.shape)
    
    return dataset


def trainTestSplit(dataset):
    
    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size
    print(train_size, test_size)
    
    datasetTrain = dataset[0:train_size]
    print('Train dataset has shape of: ', datasetTrain.shape)
    
    datasetTest = dataset[train_size:]
    print('Train dataset has shape of: ', datasetTest.shape)
    
    return datasetTrain,datasetTest


def normalizeDataset (dataset):
    
    scalar = MinMaxScaler()
    dataset_sclaed=scalar.fit_transform(dataset)
    
    return dataset_sclaed, scalar


def generateDataset (data, look_back, forward_days):
    
    x=[]
    y=[]
       
    for i in range(0,len(data) -look_back -forward_days +1, 1):
        x.append(data[i:(i+look_back)])
        y.append(data[(i+look_back):(i+look_back+forward_days)])
        
    x, y = np.array(x), np.array(y)
    
    y = np.array([list(a.ravel()) for a in y])
    
    print('shape of xtrain: ', x.shape, ' and shape of ytrain: ',  y.shape)
        
    return x, y  



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    