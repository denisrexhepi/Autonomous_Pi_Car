import cv2
import numpy as np
import glob
import sys
import time
import os
from sklearn.model_selection import train_test_split

import random
import collections

import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout, Lambda
from keras.models import load_model
from keras import backend as K

import pickle


def load_data(input_size, path):
    print("Duke ngarkuar te dhenat")
    start = time.time()

    # load training data
    X = np.empty((0, input_size))
    y = np.empty((0, 4))
    training_data = glob.glob(path)

    # if no data, exit
    if not training_data:
        print("Te dhenat nuk u gjenden")
        sys.exit()

    for single_npz in training_data:
        with np.load(single_npz) as data:
            train = data['train']
            train_labels = data['train_labels']
        X = np.vstack((X, train))
        y = np.vstack((y, train_labels))

    print("Forma e vektorit te imazheve: ", X.shape)
    print("Forma e vektorit te etiketave: ", y.shape)

    end = time.time()
    print("Ngarkimi i te dhenave zgjati: %.2fs" % (end - start))

    # normalize data
    X = X / 255.

    # train validation split, 7:3
    return train_test_split(X, y, test_size=0.2)


class NeuralNetwork(object):
    def __init__(self):
        self.model = None

    def create(self, layer_sizes):
        # create neural network
        self.model = cv2.ml.ANN_MLP_create()
        self.model.setLayerSizes(np.int32(layer_sizes))
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 0.01))

    def train(self, X, y):
        # set start time
        start = time.time()

        print("Duke u trajnuar ...")
        self.model.train(np.float32(X), cv2.ml.ROW_SAMPLE, np.float32(y))

        # set end time
        end = time.time()
        print("Kohezgjatja e trajnimit: %.2fs" % (end - start))

    def evaluate(self, X, y):
        ret, resp = self.model.predict(X)
        prediction = resp.argmax(-1)
        true_labels = y.argmax(-1)
        accuracy = np.mean(prediction == true_labels)
        return accuracy

    def save_model(self, path):
        directory = "model_i_ruajtur"
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.model.save(path)
        print("Modeli u regjistrua tek : " + "'" + path + "'")

    def load_model(self, path):
        if not os.path.exists(path):
            print("Modeli nuk ekziston, dalje")
            sys.exit()
        self.model = cv2.ml.ANN_MLP_load(path)
        print('Modeli ne OpenCV u ngarkua')
        # self.model = cv2.dnn.readNetFromTensorflow("tf_model.pb")

    def load_modelKeras(self, path):

        if not os.path.exists(path):
            print("Modeli nuk ekziston, dalje")
            sys.exit()
        self.modelKeras = load_model('model_test.h5')
        print("Modeli ne Keras u ngarkua")

    def load_modelSign(self, path):
        if not os.path.exists(path):
            print("Model does not exist, exit")
            sys.exit()
        pickle_in = open("sign_model.p", "rb")  ## rb = READ BYTE
        self.modelSign = pickle.load(pickle_in)
        print("Road sign model loaded")

    def predict(self, X):
        resp = None
        try:
            ret, resp = self.model.predict(X)
        except Exception as e:
            print(e)

        return resp.argmax(-1)

    def predictKeras(self, X):
        # model = load_model('model_test1.h5')
        X = X.reshape(X.shape[0], 120, 320,1)
        y_pred = self.modelKeras.predict_classes(X)
        # y_true = np.argmax(y_test, -1)
        # print(y_pred)
        return y_pred

    def predictSign(self, img):
        predictions = model.predict(img)
        classIndex = model.predict_classes(img)
        probabilityValue = np.amax(predictions)
        return classIndex, probabilityValue