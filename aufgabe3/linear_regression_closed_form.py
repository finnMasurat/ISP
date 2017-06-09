#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ISP 3: Lineare Regression mit abgeschlossener Lösung
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


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
        # X = [[0.86], [0.09], [-0.85], [0.87], [-0.44], [-0.43], [-1.10], [0.40], [-0.96], [0.17]]
        # np.shape(X) = (10, 1)

        # Falls Eindimensionales Array
        if len(np.shape(X)) < 2:
            X = np.reshape(X, (np.shape(X)[0], 1))           #array([[ 0.86],
                                                                    #[ 0.09],
                                                                    #[-0.85],
                                                                    #[ 0.87],
                                                                    #[-0.44],
                                                                    #[-0.43],
                                                                    #[-1.1 ],
                                                                    # [ 0.4 ],
                                                                    #[-0.96],
                                                                    #[ 0.17]])


        # TODO: X spaltenweise um x_0 = 1 (korrespondiert mit Bias-Term) ergänzen
        # damit sollte X die Form [n_samples, n_features + 1] bekommen

        xOnes = np.ones(len(X))                             #array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
        X = np.column_stack((xOnes, X))                     #array([[ 1.  ,  0.86],
                                                                   #[ 1.  ,  0.09],
                                                                   #[ 1.  , -0.85],
                                                                   #[ 1.  ,  0.87],
                                                                   #[ 1.  , -0.44],
                                                                   #[ 1.  , -0.43],
                                                                   #[ 1.  , -1.1 ],
                                                                   #[ 1.  ,  0.4 ],
                                                                   #[ 1.  , -0.96],
                                                                   #[ 1.  ,  0.17]])

        Xt = transpose(X)                                      #array([[ 1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ,  1.  ],
                                                                #      [ 0.86,  0.09, -0.85,  0.87, -0.44, -0.43, -1.1 ,  0.4 , -0.96,
                                                                #        0.17]])

        inverseXtX = inv(dot(Xt,X)) # (Xt*X)^-1                 array([[ 0.10408228,  0.02936895],
                                                                      #[ 0.02936895,  0.2112874 ]])

        # TODO: Bestimmung der Gewichte mittels abgeschlossener Lösung
        # self.weights in der Form [n_features + 1]
        self.weights = dot(dot(inverseXtX,Xt), y) # (Xt*X)^-1 *(Xt)*y    array([ 1.05881341,  1.61016842])

        return self

    def predict(self, X):
        """Vorhersage der Zielvariable

        Args:
          X: Matrix der Attribute pro Datensatz in der Form [n_samples, n_features]
        Returns:
          Array der vorhergesagten Zielvariablen in der Form [n_samples]
        """
        einsen = np.ones(np.shape(X)[0])
        einsen = np.reshape(einsen, (np.shape(einsen)[0], 1))
        X = np.concatenate((einsen, X), 1)

        # (X*sigma^T)
        return np.matmul(X, np.transpose(self.weights))
