# Introduction

American Sign Language (ASL) is a common form of communication for many people in the deaf
community around the globe who cannot speak or hear. For the millions of people worldwide who rely on
ASL, communicating with others is a major struggle, whether it be in education, a workspace, or at home.
We harnessed the capabilities of machine learning (ML) to provide an efficient and accurate way to
convert ASL to text. We aim to break the communication barrier and help bridge the gap between this
underrepresented community and the rest of the world, empowering their voices.

The machine learning algorithm of choice was a Convolutional Neural Network (CNN). A CNN is a
subfield of well-known Neural Networks that specializes in extracting image features and patterns. Its
uses are varied but mostly fall into the categories of image recognition, classification, and object
detection. This project utilized the nuances of a CNN to classify images of hand symbols as English
letters or numbers.

# Overview

The implementation of the CNN consisted of the following parts:

1. Core CNN Layers
    a. Convolution
    b. Max Pooling
    c. Fully Connected
    d. Flattening
2. Activation Functions
    a. ReLU
    b. Softmax
    c. Sigmoid
    d. Tanh
3. Loss Function
    a. Cross Entropy Loss
The usage and integration of the CNN with the ASL data consisted of the following parts:
1. Preprocessing Data
2. Save Model
a. Use Pickle Python Library
3. Evaluate and Predict

# Results

In this project, we were able to create a model with an accuracy in the high 80s and low 90s. Furthermore,
we tested possible optimizations of the model by training with different image shapes and the number of
epochs. The below chart explains our findings:


Image Shape  | Epochs   | ∼Accuracy | ∼Time
28x28        |  20      |  88%      |  30 minutes
64x64        |  20      |  93%      |  3 hours

We predict that further optimizations may include increasing the number of epochs and the dimensionality
of the image.

# Additional Instructions and Notes

Since our model does not utilize any multi-threading or GPU, the time to train each model can get quite
lengthy as shown above. Due to this, we have saved the final trained model for each variation in a pickle
file which can be loaded and used to predict on the images.
The model's creation and usage is in the ASLModel.ipynb file. The model can be retrained and tested
from this file. Finally, be sure to change the model_path, where your pickle file, storing your model, will
be saved/retreived from.
