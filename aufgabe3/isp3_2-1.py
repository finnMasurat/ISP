#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ISP 3: Lineare und logistische Regression

# Aufgabe 2.1 - Logistische Regression mit SGD

Dieses Skript beinhaltet einen Dummy-Datensatz Trainieren und Testen
eines Regressions-Modells.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
data = np.loadtxt("./data/classify.txt", delimiter=",")

X = data[:,:2]
y = data[:,2]

# Plot Dataset
plt.plot([X[i,0] for i in range(len(X)) if y[i] == 0], [X[i,1] for i in range(len(X)) if y[i] == 0], "bo")
plt.plot([X[i,0] for i in range(len(X)) if y[i] == 1], [X[i,1] for i in range(len(X)) if y[i] == 1], "ro")
plt.show()

from logistic_regression import LogisticRegression

# Training des Modells
model = LogisticRegression()
model.fit(X, y, alpha=0.01, iterations=100)

# Plot der Error-Kurve über das Training
plt.plot(model.cost, 'r')
plt.show()

# Testdaten zur Visualisierung der Decision-Boundary
xx, yy = np.mgrid[-10:10:.01, -10:10:.01]
grid = np.c_[xx.ravel(), yy.ravel()]
probs = model.predict(grid).reshape(xx.shape)

# Plot der Klassifikation mit Decision-Boundary
f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu", vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])
ax.scatter(X[:,0], X[:, 1], c=y, s=50,
           cmap="RdBu", vmin=-.2, vmax=1.2,
           edgecolor="white", linewidth=1)
ax.set(xlim=(-3.5, -0.5), ylim=(-6, 6),
       xlabel="$X_1$", ylabel="$X_2$")
plt.show()
