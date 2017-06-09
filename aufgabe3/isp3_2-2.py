#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
ISP 3: Lineare und logistische Regression

# Aufgabe 2.2: Klassifikation des Überlebens auf der Titanic

Dieses Skrip läd einen Datensatz mit Passagieren der Titanic. Das Ziel ist es,
einen Klassifikator zu Trainieren, um das Überleben des Unglücks vorher zu sagen.

Zur Vorverarbeitung werden Trainings- und Testdaten zusammengeführt. Danach kann
der Trainings-Datensatz zur Erstellung des Klassifikators verwendet werden.
Anaschließend kann die Qualität des Klassifikators anhand der Testdaten ermittelt
werden.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Laden des Datensatzes als Pandas DataFrame
training_data = pd.read_csv("./data/titanic_train.csv", sep=",")
test_data = pd.read_csv("./data/titanic_test.csv", sep=",")

# Zusammenführen der Daten zur Vorverarbeitung
merged_data = training_data.append(test_data, ignore_index=True)
print("Loaded Titanic Datset with shape {}".format(np.shape(merged_data)))

# TODO: Vorverarbeitung der Daten

# Trennen der Daten in Trainings- und Testdaten
train = merged_data[:1000]
test = merged_data[1000:]

# TODO: Training des Modells


# TODO: Testen des Modells anhand der Testdaten
