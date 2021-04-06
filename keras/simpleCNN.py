from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, losses

def build_model(width, height, depth, nb_labels):
    model = keras.models.Sequential()
    input_shape = (width, height, depth)

    coucheCv1 = keras.layers.Conv2D(32,3,input_shape=input_shape,activation='relu')
    model.add(coucheCv1)
    
    coucheMP1=keras.layers.MaxPooling2D(input_shape=input_shape,pool_size=(3,3))
    #model.add(coucheMP1)

    coucheCv2 = keras.layers.Conv2D(64,3,input_shape=input_shape,activation='relu')
    model.add(coucheCv2)

    coucheMP2=keras.layers.MaxPooling2D(input_shape=input_shape,pool_size=(3,3))
    #model.add(coucheMP2)

    model.add(keras.layers.Flatten(input_shape=input_shape))

    coucheD1 = keras.layers.Dense(256,activation='relu')
    model.add(coucheD1)

    coucheS = keras.layers.Dense(nb_labels,activation='softmax')
    model.add(coucheS)
    #Pourrait rajouter MaxPooling et Dropout

    return model