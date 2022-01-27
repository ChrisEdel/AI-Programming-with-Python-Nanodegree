# AI Programming with Python Nanodegree
This repository contains my submissions for the nanodegree program [AI Programming with Python](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) offered by [Udacity](https://www.udacity.com/).

Please note that the foundation of the code was provided by Udacity as a starting point for the projects.

## Pre-trained Image Classifier to Identify Dog Breeds

The first project dealt with using a given image classifier in order to identify dog breeds. The focus was not on training or building the classifier, but on demonstrating the necessary Python skills for setting up a machine learning project, i.e. dealing with the data, looking at different metrics, inspecting results and run times of different classifiers.

## Own Image Classifier

The second project dealt with building an image classifier almost from scratch. Only the very basic foundation of the code was given (i.e. some helper functions and tips). My submission included:
* reading and transforming the data
* choosing a suitable (pretrained) network architecture
* defining a suitable classifier for the chosen architecture
* training the neural network
* evaluating the neural network
* saving and loading checkpoints of the neural network
* illustrating the predictions visually with the corresponding probabilities

The first goal of the project was to implement the above described functionality in a Jupyter Notebook. The second goal was to build a command line application allowing the user to:
* choose different network architectures (`vgg13`, `vgg16`, `alexnet`)
* customize the hyperparameters (epochs, learning_rate, hidden units)
* use a GPU for training (if available)
* save and load the model
* use the model to make predictions

### Training
1. Checkout this repository and navigate into `image_classifier_flowers/ImageClassifier`
2. Run `python train.py <data_directory>` (the script is written in such a way that the user is informed about all the steps taken (building the network, training, ...))
3. To see all the possible customizations, run `python train.py --h`

### Predicting
1. Checkout this repository and navigate into `image_classifier_flowers/ImageClassifier`
2. Run `python predict.py <path_to_image> <checkpoint>` (note that a valid checkpoint needs to be given)
3. To see all the possible customizations, run `python predict.py --h`
