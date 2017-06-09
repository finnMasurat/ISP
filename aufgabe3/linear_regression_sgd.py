#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ISP 3: Lineare Regression mit Stochastic Gradient Descent (SGD)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

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

        # TODO: X spaltenweise um x_0 = 1 (korrespondiert mit Bias-Term) ergänzen
        # damit sollte X die Form [n_samples, n_features + 1] bekommen

        # TODO: Initialisierung der Gewichte (inklusive Bias-Term)
        # self.weights in der Form [n_features + 1]
        self.weights = …

        # Array zum Speichern des Errors für jeden SGD-Schritt
        self.cost = []

        # TODO: Hypothese h(x)
        h_x = lambda w, X: …
        # alternativ als Funktion: def h_x(w,X): …

        # TODO: Sum of Squared Errors (SSE)
        sse = lambda w, X, y: …
        # alternativ als Funktion: def sse(w,X,y): …

        # SGD Schleife über Iterationen
        for _ in range(iterations):
            # Schleife über Datensätze
            for i in range(m):

                # TODO: Update der Gewichte
                self.weights = …

                # alternativ Schleife über Parameter (n+1 durch Bias-Term)
                # for j in range(n+1):
                #   self.weights[j] = …

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
        # TODO: Berechnen der Zielvariablen über die Hypothese
        # folgende Zeile löschen
        raise NotImplementedError("Noch nicht implementiert!")
