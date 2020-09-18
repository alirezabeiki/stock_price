from dataset import *
from Networks import *
from plot import *
import tensorflow as tf
import numpy as np
import pandas as pd

'''
-------------------------------------------------------------------------------
                       Part 1: Setting Parameters
-------------------------------------------------------------------------------
'''

# Define whick market do you use.
# 1. IranStock   ;   2. Forex
stockType         = 'IranStock'

# Specified Name of the Stock.
fileName          = 'Alvand.Tile.csv'


# Define which features would be as input for trainig the model.
FeaturesInput     = ["Open", "High", "Low", "Close", "Volume"]
#FeaturesInput     = ["Close"]


# Define which features would be as ouput for trainig the model.
FeaturesOutput     = ["Close"]


# look_back: training the model by the look_back days ago.
# forward_days: predict the forward_days later.
look_back    = 60
forward_days = 10

# Epoches: Epoches & Batch_Size: Batch Size
Epoches = 100
Batch_Size=100


'''
-------------------------------------------------------------------------------
                       Part 2: Reading and Creating dataset
-------------------------------------------------------------------------------
'''

# Reading, scaling and splitting of input dataset
x= readingDataset(fileName, stockType, FeaturesInput)
x, Max_Target, Min_Target = normalize(x, FeaturesOutput)


for i in range(len(x.columns)):
    if x.columns[i] == FeaturesOutput[0]:
        _id=i
        
xtrain, xtest = trainTestSplit(x)


# Reading, scaling and splitting of output dataset
y= readingDataset(fileName, stockType, FeaturesOutput)
y, _ , _ = normalize(y, FeaturesOutput)
ytrain, ytest = trainTestSplit(y)


# Creating dataset
xtrain, ytrain = generateDataset (xtrain, ytrain, look_back, forward_days)
xtest,  ytest  = generateDataset (xtest, ytest, look_back, forward_days)



'''
-------------------------------------------------------------------------------
                       Part 3: Train the Network
-------------------------------------------------------------------------------
'''

model = LSTM1(xtrain, forward_days)
model.summary()

# Compiling the model
sgd = tf.keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
model.compile(optimizer=sgd, loss='mae')
#model.compile(optimizer='adam', loss='mse')


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

# Prediction of next forward days
# 1. Predict by model
x_pre = np.array(x[len(x)-look_back:len(x)])
x_pre = np.reshape(x_pre, (1, look_back, len(FeaturesInput)))
y_pre = model.predict(x_pre)

# 2. return to true values
x_pre = np.reshape(x_pre, (look_back, len(FeaturesInput)))
x_pre = x_pre[:, _id]
x_pre = true_scale(x_pre, Max_Target, Min_Target)

y_pre = np.reshape(y_pre, (-1,1))
y_pre = true_scale(y_pre, Max_Target, Min_Target)

y_true= readingDataset('actu.csv', stockType, FeaturesOutput)
y_true = np.array(y_true)
y_true = y_true[len(y_true)-forward_days:len(y_true)]
#y_true = y_true[len(y_true)-forward_days:len(y_true)]

plot_prediction(x_pre, y_pre, y_true, look_back, forward_days)








