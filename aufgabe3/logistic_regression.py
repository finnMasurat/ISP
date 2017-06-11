#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ISP 3: Logistische Regression mit Stochastic Gradient Descent (SGD)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class LogisticRegression(object):
    """Logistic Regression Model

    Diese Klasse trainiert ein logistisches Regressionsmodell mittels SGD
    """

    def __init__(self, random_seed=1):
        self.random_seed = random_seed


    def fit(self, X, y, alpha=0.001, iterations=1000):
        """Training des Modells anhand gegebener Trainingsdaten

        Args:
          X: Matrix der Trainingsdaten in der Form [n_samples, n_features]
          y: Vektor der Zielvariablen in der Form [n_samples]
          alpha: Lernrate des Gradientenverfahrens
          iterations: Anzahl der Iterationen von SGD über die Daten X

        Returns:
          self: Modell mit trainierten Gewichten
        """
        # Erweitern der Dimension von X falls X nur ein Feature beinhaltet
        # [n_samples] -> [n_samples, 1]
        if len(np.shape(X)) < 2:
            X = np.reshape(X, (np.shape(X)[0], 1))

        # m Samples mit n Attributen (exclusive Bias)
        m, n = np.shape(X)

        # self.weights in der Form [n_features + 1]
        #Drei zufällige Gewichte zwischen -3 und 3 bestimmen und self.weights mit diesen initialisieren
        random1 = np.random.uniform(-3,3)
        random2 = np.random.uniform(-3,3)
        random3 = np.random.uniform(-3,3)
        self.weights = np.array([[random1],[random2],[random3]])

        # Array zum Speichern des Errors für jeden SGD-Schritt
        self.cost = []

        # append attribute x_0 to training samples
        x_0 = np.ones((m,1))
        X = np.concatenate([x_0, X], axis=1)

        def sigmoid(x):
            return (1 / (1+np.exp(-x)))

        def h_x(w,X):
            return sigmoid(np.matmul(np.transpose(w),X))

        def cost_fn(w,X,y):
            sum = 0
            for i in range(1,m):
                sum = np.add(sum, y[i] * np.log(h_x(w, X[i])) + (1-y[i]) * np.log(1-h_x(w, X[i])))
            return -sum/m


        # SGD Schleife über Iterationen
        for _ in range(iterations):
            # Schleife über Datensätze
            for i in range(m):

                # Update der Gewichte
                we = self.weights

                term = y[i] - h_x(we, X[i,])
                self.weights = we + np.multiply(alpha,np.transpose(np.multiply(term, np.array([X[i,]]))))

            # Berechne SSE des SGD-Schritts mit den aktuallisierten Gewichten
            self.cost = self.cost + [cost_fn(self.weights, X, y)]
        print(self.weights)
        return self

    def predict(self, X):
        """Vorhersage der Zielvariable

        Args:
          X: Matrix der Attribute pro Datensatz in der Form [n_samples, n_features]
        Returns:
          Array der vorhergesagten Zielvariablen in der Form [n_samples]
        """
        def sigmoid(x):
            return (1 / (1+np.exp(-x)))
        return sigmoid(np.dot(X, self.weights[1:]) + self.weights[0])
