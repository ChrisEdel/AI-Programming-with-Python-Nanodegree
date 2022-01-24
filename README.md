# AI Programming with Python Nanodegree
This repository contains my submissions for the nanodegree program [AI Programming with Python](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) offered by [Udacity](https://www.udacity.com/).

Please note that the foundation of the code was provided by Udacity as a starting point for the projects.

## Pre-trained Image Classifier to Identify Dog Breeds

The first project dealt with using a giving image classifier in order to identify dog breeds. The focus was not on training or building the classifier, but more on demonstration the necessary Python skills for setting up a project like this, i.e. dealing with the data, looking at different metrics, inspecting results and run times of different classifiers.

## Own Image Classifier

The first project dealt building an image classifier from scratch with only the very basic foundation of the given (i.e. some helper functions and tips). The submission included:
* reading and transforming the data
* choosing a suitable (pretrained) network architecture
* defining a suitable classifier for the chosen architecture
* training the neural network
* evaluating the neural network
* saving and loading checkpoints of the neural network
* illustrating the predictions visually with the corresponding probabilities

The first goal of the project was to implement to above described functionality in a Jupyter Notebook. The second goal was to build a command line application allowing the user to:
* choose different network architectures (`vgg13`, `vgg16`, `alexnet')
* customize the hyperparameters (epochs, learning_rate)
* save and load the model
* use the model to make predictions
