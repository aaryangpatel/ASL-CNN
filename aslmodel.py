import numpy as np
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Model.Network import Network
from Layers.MaxPooling import MaxPooling
from Layers.FullyConnected import FullyConnected
from Layers.Convolution import Convolution
from Loss.CrossEntropyLossFunction import CrossEntropyLossFunction
from Activation.Sigmoid import Sigmoid
from Activation.Softmax import Softmax
from Activation.Tanh import Tanh
from Layers.Flatten import Flatten
from Activation.ReLU import ReLU
images = []
labels = []

for folder in os.listdir('asl_dataset'):
    folder_dir = os.path.join('asl_dataset', folder)

    for image in os.listdir(folder_dir):
        img_path = os.path.join(folder_dir, image)

        data_sample = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        data_sample = cv2.resize(data_sample, (64, 64))
        data_sample = data_sample / 255.0

        images.append(data_sample)
        labels.append(folder)
df = pd.DataFrame(labels)
df = pd.get_dummies(df)
x_data = np.array(images).reshape(-1, 1, 64, 64)
y_data = df.values
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
loss_function = CrossEntropyLossFunction()

network_layers = [
    Convolution(input_shape=(1, 64, 64), output_depth=16, kernel_size=3),
    ReLU(),
    MaxPooling(pool_size=2, stride=2),

    Convolution(input_shape=(16, 31, 31), output_depth=32, kernel_size=3),
    ReLU(),
    MaxPooling(pool_size=2, stride=2),

    Convolution(input_shape=(32, 14, 14), output_depth=64, kernel_size=3),
    ReLU(),
    MaxPooling(pool_size=2, stride=2),

    Flatten(),

    FullyConnected(input_size=64 * 6 * 6, output_size=256),
    ReLU(),

    FullyConnected(input_size=256, output_size=y_data.shape[1]),
    Softmax()
]
print("hi")
ASL_model = Network(network_layers, loss_function)
ASL_model.train(x_train, y_train, 20, 0.01)

import pickle

filename = 'ASL_traineddd_model.pkl'

with open(filename, 'wb') as file:
    pickle.dump(ASL_model, file)
