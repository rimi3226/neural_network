import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Data")))

import activationFunc as af
from Node import Node 
from Layer import Layer
from NeuralNetwork import NeuralNetwork
from data import preprocess_data

class Train:
    def __init__(self, epoch, batch, activation, output_activation, weight_init, optimizer, dropout, learning_rate):
        self.epoch = epoch
        self.batch = batch
        self.activation = activation
        self.output_activation = output_activation
        self.weight_init = weight_init
        self.optimizer = optimizer
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.model = None
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    def load_data(self, csv_path):
        inputs, targets_one_hot, targets = preprocess_data(csv_path)
        return inputs, targets_one_hot, targets

    def k_fold_split(self, X, y, k=5):
        fold_size = len(X) // k
        folds = []
        for i in range(k):
            X_val = X[i * fold_size:(i + 1) * fold_size]
            y_val = y[i * fold_size:(i + 1) * fold_size]
            X_train = np.concatenate((X[:i * fold_size], X[(i + 1) * fold_size:]), axis=0)
            y_train = np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]), axis=0)
            folds.append((X_train, y_train, X_val, y_val))
        return folds

    def train(self, X_train, y_train, X_val, y_val):
        input_size = 784
        output_size = 154
        hidden_layers = [256, 200]

        self.model = NeuralNetwork(
            input_layer_size=input_size,
            hidden_layers_size=hidden_layers,
            output_layer_size=output_size,
            learning_rate=self.learning_rate,
            activation=self.activation,
            output_activation=self.output_activation,
            weight_init=self.weight_init,
            optimizer=self.optimizer,
            dropout=self.dropout,
        )

        for epoch in range(self.epoch):
            for i in range(0, len(X_train), self.batch):
                X_batch = X_train[i:i + self.batch]
                y_batch = y_train[i:i + self.batch]
                for x, y in zip(X_batch, y_batch):
                    self.model.set_input_layer(x)
                    self.model.set_target_layer(np.argmax(y))
                    self.model.forward_propagation()
                    self.model.backward_propagation()

            train_loss, train_accuracy = self.evaluate(X_train, y_train)
            val_loss, val_accuracy = self.evaluate(X_val, y_val)

            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_accuracy)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            print(f"Epoch {epoch + 1}/{self.epoch}, Loss: {train_loss}, Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

    def evaluate(self, X, y):
        losses = []
        correct = 0
        for x, target in zip(X, y):
            self.model.set_input_layer(x)
            self.model.forward_propagation()
            prediction = np.argmax([node.val for node in self.model.layers[-1].nodes])
            losses.append(np.sum((target - prediction) ** 2))  # MSE Loss
            correct += (prediction == np.argmax(target))
        return np.mean(losses), correct / len(X)

    def save_results(self, results_path="results.csv", weights_path="weights.csv"):
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(results_path, index=False)
        self.model.store_weight(weights_path)

    def plot_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.history["loss"], label="Training Loss")
        plt.plot(self.history["val_loss"], label="Validation Loss")
        plt.plot(self.history["accuracy"], label="Training Accuracy")
        plt.plot(self.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training History")
        plt.xlabel("Epochs")
        plt.ylabel("Value")
        plt.legend()
        plt.grid()
        plt.show()
