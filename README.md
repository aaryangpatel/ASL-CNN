**CNN Implementation For ASL Detection**

Team Members: Aaryan Patel and Peter Morand

Due Date: March 5

**Introduction:**

**American Sign Language** (ASL) is a common form of communication for many people in the deaf community around the globe who cannot speak or hear. For the millions of people worldwide who rely on ASL, communicating with others is a major struggle, whether it be in education, a workspace, or at home. We propose to harness the capabilities of machine learning (ML) to provide an efficient and accurate way to convert ASL to text. We aim to break the communication barrier and help bridge the gap between this underrepresented community and the rest of the world, empowering their voices.

The machine learning algorithm of choice is a **Convolutional Neural Network** (CNN). A CNN is a subfield of well-known **Neural Networks** that specializes in extracting image features and patterns. Its uses are varied but mostly fall into the categories of image recognition, classification, and object detection. This project will utilize the nuances of a CNN to classify images of hand symbols as English letters or numbers in real time.

**Overview:**

The core aspects of the project are as follows:

1.  Ground-up design, implementation, and optimization of the core functionality of a CNN.
    
2.  Train a model using a robust ASL dataset to detect letters and numbers from pictures of individual hand symbols.
    
3.  Develop an application to take a picture of a hand gesture and return the corresponding text in real-time.
    
4.  If time permits, enhance the application to allow video input to convert long-form ASL to text.
    

**Project Objectives:**

Alongside the overarching goal of this project to help the deaf community communicate, this project will allow us to uncover various technical details, optimizations, and uses of the machine learning algorithms applied. This project has the following objectives:

1.  Application that assists with ASL translation of characters from images.
    
2.  Implementation of a CNN.
    
    1.  This will allow us to demystify the "black box" of CNNs and discover how to optimize their performance. 
        
    2.  This will also enable us to better understand how these ML algorithms are used in the real world and their pros and cons.
        
3.  Use real time camera images instead of pre-taken photos from the dataset.
    
    1.  This will have real time conversion.
        
    2.  This will allow an easier transition to detecting ASL in videos with consecutive frames of moving symbols rather than still images.
        

**Methodology:**

1.  Research the details of the core aspects of a CNN.
    
2.  Implementation of a CNN in Python
    
    1.  **Convolutional Layers:** The first and most important layer of a CNN, which takes in an image and extracts key features.
        
    2.  **Pooling Layers:** Combines neuron output from the convolution stage.
        
    3.  **Fully Connected Layers:** Analogous to a multilayer perceptron, which returns a probability distribution for each class.
        
    4.  **Backpropagation and Gradient Descent:** Process to train the weight of the neural network.
        
    5.  Use **NumPy** and **Matplotlib**
        
3.  Data Collection and Preprocessing
    
    1.  Data: [https://www.kaggle.com/datasets/ayuraj/asl-dataset](https://www.kaggle.com/datasets/ayuraj/asl-dataset)
        
    2.  Preprocess data (images) to work with the developed model.
        
    3.  Using **Matplotlib** and **OpenCV**
        
4.  Train and Evaluate the Model
    
    1.  Utilize the built model framework and dataset to train a working model.
        
    2.  Use different metrics to assess the accuracy of the model.
        
    3.  Use hyperparameters or optimizations in the model's development to enhance performance.
        
5.  Real-Time Imaging
    
    1.  Extract the symbol from an image taken from a camera.
        
    2.  Preprocess the data.
        
    3.  Utilize the trained model to predict the character.
        
    4.  Using **OpenCV**
        
6.  Video-To-Text (If Time Permits)
    
    1.  Frame extraction and preprocessing.
        
    2.  Utilize the trained model to predict long-form text.
        
    3.  Using **OpenCV**
