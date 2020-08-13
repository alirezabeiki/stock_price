import numpy as np
import matplotlib.pyplot as plt


def historyPlot(history):
    
    plt.figure(figsize = (15,10))
    
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(loc='best')
    plt.show()
    
    return print('Plot history')

def plot_prediction_target(Xt, forward_days, ytest, sc):
    
    plt.figure(figsize = (15,10))
    
    for i in range(0,len(Xt)):
        plt.plot([x + i*forward_days for x in range(len(Xt[i]))], sc.inverse_transform(Xt[i].reshape(-1,1)), color='r')
    
    plt.plot(0, sc.inverse_transform(Xt[i].reshape(-1,1))[0], color='r', label='Prediction') #only to place the label
        
    plt.plot(sc.inverse_transform(ytest.reshape(-1,1)), label='Target')
    plt.legend(loc='best')
    plt.show()
    
    return


def plot_Future_Price(y, forward_days, price):
    
    plt.figure(figsize = (15,10))
    
    x=[]
    for i in range(forward_days):
        x.append(i)
    
    plt.plot(x,y, label='Prediction')
    plt.plot(x,price, label='True Price')
    plt.legend(loc='best')
    plt.show()
    
    return
    
def plot_Predict(Q, W, forward_days):
    
    plt.figure(figsize = (15,10))
    
    x=[]
    for i in range(forward_days):
        x.append(i)
        
    y=[]
    for i in range(forward_days):
        y.append(forward_days+i)
    
    plt.plot(x,Q[:,0], label='Look Back')
    plt.plot(y,W[:,0], label='Forward Days')
    plt.legend(loc='best')
    plt.show()
    
    return
    
    
