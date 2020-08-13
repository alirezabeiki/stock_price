import tensorflow as tf


def LSTM1(xtrain, forward_days):
    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(50, activation='relu', input_shape=(xtrain.shape[1],xtrain.shape[2]),return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.LSTM(60, activation='relu', return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.LSTM(80, activation='relu', return_sequences = True))
    model.add(tf.keras.layers.Dropout(0.4))
    
    model.add(tf.keras.layers.LSTM(120, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(tf.keras.layers.Dense(forward_days,activation='linear'))
    
    return model

