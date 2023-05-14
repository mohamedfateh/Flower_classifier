# Flower_classifier
Flower Classification with Deep Neural Networks
This is a computer vision project that uses deep neural networks to classify different types of flowers based on their images.
The project uses TensorFlow and Python programming language to build and train a convolutional neural network (CNN) model for image classification.

Dataset:
The dataset used for this project is obtained from Kaggle. It consists of images of five different types of flowers:

Daisy
Dandelion
Rose
Sunflower
Tulip
The dataset contains 4,050 images, with each class having 810 images. The images are in RGB format and have varying sizes.

Methodology
The CNN model used for this project is a four-layer architecture consisting of a convolutional layer, a max-pooling layer, a flattening layer, and a dense output layer. The ReLU activation function is used in the convolutional and dense layers, and dropout regularization is applied to reduce overfitting.

The model is trained using the Adam optimizer and cross-entropy loss function, with a learning rate of 0.001 and a batch size of 32. The training is stopped after 20 epochs to prevent overfitting.

Results
The trained model achieves an accuracy of 89.67% on the test dataset, with a precision of 90.17%, recall of 89.67%, and an F1 score of 89.92%.

Requirements
The project requires the following Python libraries:

TensorFlow 2.4.1
NumPy 1.19.5
Matplotlib 3.3.4
Scikit-learn 0.24.1
Usage
To use this project, you can follow these steps:

Clone this repository to your local machine.
Install the required libraries using pip install -r requirements.txt.
Download the dataset from Kaggle and extract it to the ./data directory.
Run python train.py to train the model.
Run python evaluate.py to evaluate the model on the test dataset.
Run python predict.py path/to/image.jpg to predict the class of a new image.
License
This project is licensed under the MIT License. See the LICENSE file for more information.

Acknowledgments
Kaggle for providing the dataset used in this project
