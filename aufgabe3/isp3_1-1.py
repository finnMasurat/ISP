#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ISP 3: Lineare und logistische Regression

# Aufgabe 1.1 - Abgeschlossene Lösung

Dieses Skript beinhaltet einen Dummy-Datensatz zum Trainieren und Testen
eines Regressions-Modells.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

# Dummy Datensatz mit m=10 Samples und n=1 Attribute
X = [[0.86], [0.09], [-0.85], [0.87], [-0.44], [-0.43], [-1.10], [0.40], [-0.96], [0.17]]
y = [2.49, 0.83, -0.25, 3.10, 0.87, 0.02, -0.12, 1.81, -0.83, 0.43]

# Testdaten
X_test = np.arange(-2,2,0.1)
X_test = np.reshape(X_test, (np.shape(X_test)[0],1))


from linear_regression_closed_form import LinearRegression

# Training des Modells
model = LinearRegression()
print(X)
model = model.fit(X,y)

# Plot der Trainingsdaten und der Annäherung
plt.plot(X, y, 'bo')
plt.plot(X_test, model.predict(X_test), 'r--')
plt.show()
