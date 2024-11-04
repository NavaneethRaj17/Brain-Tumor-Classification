# Brain Tumor Classification
Brain tumors can be devastating, and in India, the situation is particularly challenging due to a shortage of access to oncologists and neurosurgeons. The severity of brain tumors varies, with some being benign and others malignant, which can significantly impact treatment options and outcomes. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and 36 percent for women.
There are several types of brain tumors, including:
* Benign tumors, which are non-cancerous and typically don't spread to other parts of the brain
* Malignant tumors, which are cancerous and can spread to other parts of the brain and spinal cord

# Different types of cancerous brain tumors
1. # Meningioma
   Meningioma is a type of tumor that develops in the membranes surrounding the brain and spinal cord, with most being benign but some being malignant.

2. # Glioma
   Glioma is a type of brain tumor originating from brain tissue, classified into low-grade (slow-growing) and high-grade (aggressive) categories.

3. # Pituitary Tumors
   Pituitary tumors develop in the pituitary gland, with most being benign adenomas and some being malignant carcinomas. Symptoms result from hormonal imbalances, including vision loss, headaches, infertility, and Cushing's syndrome.

# Convolutional Neural Networks (CNNs) in Radiology
The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by the radiologist. A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has consistently shown higher accuracy than manual classification. Hence, proposing a system performing detection and classification by using Deep Learning Algorithms using ConvolutionNeural Network (CNN), Artificial Neural Network (ANN), and TransferLearning (TL) would be helpful to doctors all around the world.
Convolutional Neural Networks (CNNs) have revolutionized the field of radiology by providing a powerful tool for analyzing medical images. CNNs are particularly well-suited for image recognition and classification tasks:
* Image Segmentation
* Disease Detection and Diagnosis
* Image Reconstruction
* Image Registration

# Classifying brain tumors using MRI images with a custom Convolutional Neural Network (CNN)

# Dataset
Brain tumor classification dataset from [Kaggle](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri). The dataset consists of four classes: glioma, meningioma, notumor, and pituitary. The images are already split into Training and Testing folders. Each folder has four more subfolders. These folders have MRIs of respective tumor classes.

<img width="391" alt="training_data" src="https://github.com/user-attachments/assets/52a3e978-c43a-49fe-a1ae-3dd28a6cdc94">




<img width="388" alt="testing_data" src="https://github.com/user-attachments/assets/a78e9ce1-b838-4763-80e0-0bb7879265c6">


# Kaggle API configuration for a Google Colab environment
* Set the Kaggle config directory: You specify it as /content, which is the root directory in Colab.
* Create the .kaggle directory: You're creating it in the home directory (~/.kaggle) where kaggle.json is usually stored.
* Copy the Kaggle API key file: You copy kaggle.json into the .kaggle folder.
* Ensure that the kaggle.json file is available in /content before running the python script.
* run Kaggle commands ! kaggle datasets download sartajbhuvaji/brain-tumor-classification-mri

# Install all the dependencies
* import numpy as np
* import tensorflow as tf # The core library for deep learning, with Sequential and layers modules for building and training the model.
* from tensorflow.keras.models import Sequential
* from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
* from tensorflow.keras.preprocessing.image import ImageDataGenerator # For augmenting and loading images, helping to prevent overfitting by increasing data variability.
* from tensorflow.keras.optimizers import Adam # A widely-used optimizer in deep learning for adaptive learning rates.
* from tensorflow.keras.preprocessing import image # To load individual images for predictions or visualizations.
* from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # For performance metrics, helping to evaluate model accuracy and detailed classification performance.
* import pandas as pd
* import numpy as np
* from PIL import Image # For working with images directly.
* import seaborn as sns
* import matplotlib.pyplot as plt
* import warnings # To ignore warnings during model training, which can often be expected but aren’t critical.
* import zipfile

# Data augmentation for the training images
Improve the model's generalization by creating slightly varied versions of each image.ImageDataGenerator is a class in TensorFlow/Keras that provides a powerful way to augment images on-the-fly during training:
* Rotation: Randomly rotates images by a specified degree range.
* Shifting: Randomly shifts images horizontally or vertically.
* Zooming: Randomly zooms images in or out.
* Shear: Randomly shears images.
* Flipping: Randomly flips images horizontally or vertically.
* Channel Shifting: Randomly shifts the intensity of each color channel.
* Rescaling: Rescales the image pixel values to a specific range (e.g., 0-1).
  
Benefits of Data Augmentation:
* Increased Dataset Size: By generating new, slightly modified images from existing ones, you can effectively increase the size of your training dataset.
* Improved Generalization: Exposing your model to a wider range of image variations helps it generalize better to unseen data.
* Reduced Overfitting: By making the model more robust to variations in input data, data augmentation can help prevent overfitting.
By leveraging ImageDataGenerator, you can significantly enhance the performance of your image classification, object detection, and other computer vision models, especially when working with limited datasets.

# Label Encoding
One-hot encoding is a technique used to represent categorical data as binary vectors. It’s commonly used in machine learning and deep learning for representing labels in a format that models can understand, especially in classification problems.In one-hot encoding, each category (or class) in the data is represented by a vector that has a length equal to the number of categories. All values in the vector are set to 0 except for the index that corresponds to the category, which is set to 1.
Advantages of one hot encoding:
1. Non-Ordinal Representation: Many machine learning algorithms assume numerical values have an order (e.g., 0, 1, 2). One-hot encoding prevents the model from assuming any inherent order among categories.
2. Compatibility with Neural Networks: For classification problems, neural networks expect the target labels to be presented as one-hot encoded vectors when using a categorical loss function like categorical_crossentropy.
3. Multi-Class Classification: One-hot encoding is essential for multi-class classification tasks, where the model outputs a probability distribution across all classes.
In Keras, when you set class_mode='categorical' in ImageDataGenerator, it automatically one-hot encodes the labels for you.

# CNN model architecture
* Input Layer:
input_shape=(IMG_WIDTH, IMG_HEIGHT, 3): Specifies the shape of the input images. Here, 3 indicates that the images are RGB (3 color channels).
* Convolutional Layers:
  * Conv2D(32, (3, 3), activation='relu'): The first convolutional layer applies 32 filters of size 3x3, using the ReLU activation function.
  * MaxPooling2D((2, 2)): This layer downsamples the feature maps from the previous layer by taking the maximum value over a 2x2 window, reducing the spatial dimensions.
  * Additional Convolutional Layers: You have two more convolutional layers (64 and 128 filters) with ReLU activation, each followed by max pooling. This hierarchy helps the model learn increasingly complex features from the images.
* Flattening Layer:
Flatten(): Converts the 3D output from the final convolutional layer into a 1D vector, making it suitable for input into the dense layers.
* Fully Connected (Dense) Layers:
  * Dense(128, activation='relu'): A dense layer with 128 neurons and ReLU activation, which helps in learning complex representations.
  * Dropout(0.2): A dropout layer that randomly sets 20% of the inputs to 0 during training, which helps prevent overfitting.
* Output Layer:
Dense(4, activation='softmax'): The final layer has 4 units (one for each class) and uses the softmax activation function, which outputs a probability distribution across the classes.

# Interpretation of Results
The CNN model for classification of brain tumors from MRI images achieved an accuracy of 85%

![Screenshot 2024-11-04 153053](https://github.com/user-attachments/assets/9807e9b9-5ef1-4e4d-a647-5b123482c5e6)



![Screenshot 2024-11-04 153137](https://github.com/user-attachments/assets/93e119c6-0749-4e76-9fec-1be828e8d02b)





    






