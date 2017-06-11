#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ISP 3: Lineare Regression mit Stochastic Gradient Descent (SGD)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import dot, transpose
from numpy.linalg import inv

class LinearRegression(object):
    """Linear Regression Model

    Diese Klasse trainiert ein lineares Regressionsmodell mittels SGD
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

        # damit sollte X die Form [n_samples, n_features + 1] bekommen
        ones = np.ones(len(X))
        X = np.column_stack((ones, X))
        Xt = transpose(X)

        self.weights = np.array([np.random.uniform(-3, 3) for _ in range(n+1)])

        # Array zum Speichern des Errors für jeden SGD-Schritt
        self.cost = []

        h_x = lambda w, X: dot(X,w)

        def sse(w,X,y):
            current_sum = 0
            for i in range(1, m):
                current_sum = np.add(current_sum, np.power(np.subtract(h_x(w, X[i]),y[i]),2))
            return 0.5*current_sum

        # SGD Schleife über Iterationen
        for _ in range(iterations):
            # Schleife über Datensätze
            for i in range(m):

                # term = y[i] - (t(w) %*% X[i,])
                term = y[i] - np.matmul(X[i,], self.weights)

                self.weights += alpha * (np.transpose(term) * X[i,])

            # Berechne SSE des SGD-Schritts mit den aktuallisierten Gewichten
            self.cost = self.cost + [sse(self.weights, X, y)]

        return self

    def predict(self, X):
        """Vorhersage der Zielvariable

        Args:
          X: Matrix der Attribute pro Datensatz in der Form [n_samples, n_features]
        Returns:
          Array der vorhergesagten Zielvariablen in der Form [n_samples]
        """

        # Bias in X einfügen
        einsen = np.ones(np.shape(X)[0])
        einsen = np.reshape(einsen, (np.shape(einsen)[0], 1))
        X = np.concatenate((einsen, X), 1)

        # Formel für Hypothese bei linearer Regression (X*sigma^T)
        return np.matmul(X, self.weights)
