import numpy as np
import os
from Layer_Dense import Layer_Dense
from Activation_Relu import Activation_ReLU
from LossFunction import Loss_CategoricalCrossentropy
from Softmax import Activation_Softmax
from Optimizer import Optimizer_SGD
from tensorflow import keras as data
from PIL import Image
from DigitRecGUI import DigitRecGUI
import tkinter as tk
class Model:
    def __init__(self, layers, batch_size, epochs):
        self.layers = [Layer_Dense(*layer) for layer in layers]
        self.activations = [Activation_ReLU() if i < len(layers) - 1 else Activation_Softmax() for i in range(len(layers))]
        self.loss_function = Loss_CategoricalCrossentropy()
        self.optimizer = Optimizer_SGD()
        self.batch_size = batch_size
        self.epochs = epochs

    def load_data(self, data):
        (X_train, Y_train), (X_test, Y_test) = data
        X_train, X_test = X_train / 255.0, X_test / 255.0
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
        self.X_train, self.Y_train, self.X_test, self.Y_test = X_train, Y_train, X_test, Y_test

    def train(self):
      # Number of batches
      n_batches = int(np.ceil(self.X_train.shape[0] / self.batch_size))

      for epoch in range(self.epochs):
          # Initialize list to store batch losses
          batch_losses = []

          for i in range(n_batches):
              # Get batch data
              start = i * self.batch_size
              end = start + self.batch_size
              X_batch = self.X_train[start:end]
              y_batch = self.Y_train[start:end]

              # One-hot encode y_batch
              y_batch_one_hot = np.eye(10)[y_batch]

              # Forward pass
              for layer, activation in zip(self.layers, self.activations):
                  layer.forward(X_batch)
                  activation.forward(layer.output)
                  X_batch = activation.output

              # Compute loss
              loss = self.loss_function.forward(activation.output, y_batch_one_hot)
              batch_losses.append(loss)

              # Backward pass
              self.loss_function.backward(activation.output, y_batch_one_hot)
              dvalues = self.loss_function.dinputs
              for layer, activation in reversed(list(zip(self.layers, self.activations))):
                  activation.backward(dvalues)
                  layer.backward(activation.dinputs)
                  dvalues = layer.dinputs

                  # Update weights and biases
                  self.optimizer.update_params(layer)

        # Compute and print average loss for this epoch
          avg_loss = np.mean(batch_losses)
          print(f'Epoch {epoch+1}, average loss: {avg_loss}')


    def predict(self, X):
        # Forward pass
        for layer, activation in zip(self.layers, self.activations):
            layer.forward(X)
            activation.forward(layer.output)
            X = activation.output

        # Compute predictions
        predictions = np.argmax(activation.output, axis=1)
        return predictions
    def save_model(self, version):
        save_dir = 'Models/V' + str(version)
        os.makedirs(f'Models', exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        for i, layer in enumerate(self.layers):
            layer.save(save_dir + '/v' + str(version) + '_layer' + str(i+1) + '.npz')

    def load_model(self, version):
        load_dir = 'Models/V' + str(version) + '/'

        for i, layer in enumerate(self.layers):
            layer.load(load_dir + 'v' + str(version) + '_layer' + str(i+1) + '.npz')

    def evaluate(self, X_val, y_val):
        for layer, activation in zip(self.layers, self.activations):
            layer.forward(X_val)
            activation.forward(layer.output)
            X_val = activation.output

                # Compute predictions
            predictions = np.argmax(activation.output, axis=1)

                # Calculate accuracy
            accuracy = np.mean(predictions == y_val)
        print(f'Validation accuracy: {accuracy * 100}%')
    def load_TrainData(self, filename):
        # Load the data from the npz file
        data = np.load('data.npz')

        # Access the images and labels
        X_train = data['images']
        Y_train = data['labels']

        print("Shapes of loaded images:", [img.shape for img in X_train])

        # Normalize the images
        X_train = X_train / 255.0

        # Reshape the images
        X_train = X_train.reshape(X_train.shape[0], -1)

        self.X_train, self.Y_train = X_train.astype(np.float32), Y_train.astype(np.int32)
