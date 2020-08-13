from dataset import *
from Networks import *
from plot import *

import numpy as np


'''
-------------------------------------------------------------------------------
                       Part 1: Setting Parameters
-------------------------------------------------------------------------------
'''


fileName     = './data/EURUSD=X.csv'
#fileName_Original     = './data/EURUSD=X.csv'
stockType    = 'Forex'        #'Forex, IranStock'
look_back    = 60
forward_days = 30
Epoches = 50
Batch_Size=50




'''
-------------------------------------------------------------------------------
                       Part 2: Reading and Creating dataset
-------------------------------------------------------------------------------
'''

dataset = closePrice(fileName, stockType)
datasetTrain, datasetTest = trainTestSplit(dataset)
datasetTrain_Scaled, scalar_Train = normalizeDataset(datasetTrain)
xtrain, ytrain = generateDataset(datasetTrain_Scaled, look_back, forward_days)

datasetTest_Scaled, scalar_Test = normalizeDataset(datasetTest)
xtest, ytest = generateDataset(datasetTest_Scaled, look_back, forward_days)



'''
-------------------------------------------------------------------------------
                       Part 3: Train the Network
-------------------------------------------------------------------------------
'''

model = LSTM1(xtrain, forward_days)
model.summary()

# Compiling the model
model.compile(optimizer='adam', loss='mse')

# Train the Model
history=model.fit(xtrain, 
                  ytrain, 
                  epochs=Epoches, 
                  batch_size=Batch_Size, 
                  verbose=2,
                  shuffle=True,
                  validation_data=(xtest, ytest))

historyPlot(history)


'''
-------------------------------------------------------------------------------
                       Part 4: Prediction
-------------------------------------------------------------------------------
'''

Y = model.predict(xtest)
plot_prediction_target(Y, forward_days, ytest, scalar_Test)


'''
dataTrue = closePrice(fileName_Original, stockType)
dataTrue = np.array(dataTrue)
dataTrueF=dataTrue[len(dataTrue)-forward_days:len(dataTrue),:]
'''

# Get last look_bak days and predict forward_days of future.
datasetFuture=dataset[(len(dataset)-look_back):len(dataset)]
datasetFuture_Scaled, scalar_Future = normalizeDataset(datasetFuture)
datasetFuture_Scaled=np.reshape(datasetFuture_Scaled,(1,look_back,1))
Future_Price = model.predict(datasetFuture_Scaled)
Future_Price=np.reshape(Future_Price,(forward_days,1))
Future_Price=scalar_Future.inverse_transform(Future_Price)


plot_Future_Price(Future_Price, forward_days, dataTrueF)



data=Y[len(Y)-1,:]
data=np.reshape(data,(len(data),1))
data=scalar_Test.inverse_transform(data)

plot_Predict(data, Future_Price, forward_days

             

             

             

      