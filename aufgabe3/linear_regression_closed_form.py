#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ISP 3: Lineare Regression mit abgeschlossener Lösung
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy import dot, transpose
from numpy.linalg import inv


class LinearRegression(object):
    """Linear Regression Model

    Diese Klasse trainiert ein lineares Regressionsmodell mittels abgeschlossener Lösung
    """

    def fit(self, X, y):
        """Training des Modells anhand gegebener Trainingsdaten

        Args:
          X: Matrix der Trainingsdaten in der Form [n_samples, n_features]
          y: Vektor der Zielvariablen in der Form [n_samples]

        Returns:
          self: Modell mit trainierten Gewichten
        """
        # Erweitern der Dimension von X falls X nur ein Feature beinhaltet
        # [n_samples] -> [n_samples, 1]
        if len(np.shape(X)) < 2:
            X = np.reshape(X, (np.shape(X)[0], 1))

        # damit sollte X die Form [[1, 1, 1, 1, 1 ...][n_samples]] bekommen
        ones = np.ones(len(X))
        X = np.column_stack((ones, X))
        Xt = transpose(X)

        # (X*Xt)^-1*Xt*y = closed form
        product = dot(Xt, X)
        theInverse = inv(product)
        self.weights = dot(dot(theInverse, Xt), y)

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
        return np.matmul(X, np.transpose(self.weights))
