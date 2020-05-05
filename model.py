import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_parser import load_data

def make_network():
    """
    Create the neural network using Keras, returns an untrained network
    """
    my_net = tf.keras.models.Sequential()

    my_net.add(tf.keras.layers.Dense(15, input_shape = (15,), activation = 'relu'))

    my_net.add(tf.keras.layers.Dense(30, activation = 'relu'))

    my_net.add(tf.keras.layers.Dense(50, activation = 'relu'))

    my_net.add(tf.keras.layers.Dense(30, activation = 'relu'))

    my_net.add(tf.keras.layers.Dense(15, activation = 'relu'))

    my_net.add(tf.keras.layers.Dense(10, activation = None))
    my_net.add(tf.keras.layers.Flatten())

    my_net.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    my_net.compile(loss = 'binary_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
    return my_net

def analyze_data(model, X, y):
    """
    args:
        model - Keras sequential model
        X - features numpy array (N_samples, N_features)
        y - labels (N_samples, 1)

    returns: 
        trained model
        two plots:
            - Accuracy v. Epochs
            - Loss v. Epochs
    """
    history = model.fit(X, y, epochs = 200, validation_split=0.4)
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    return model

def __main__():
    """
    Creates, trains, and saves a neural network using the stored data
    """
    model = make_network()

    X, y = load_data("pres12.pkl")
    Xp, yp = load_data("pres16.pkl")
    X = np.concatenate((X, Xp), axis = 0)
    y = np.concatenate((y, yp), axis = 0)

    new_model = analyze_data(model, X, y)
    new_model.save('my_new_model.h5')