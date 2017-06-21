import keras
from keras.layers import Dropout
from keras.models import Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers import Dense

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 100

# Laden des Datensatzes und Aufteilung in 60.000 Trainings- und 10.000 Testdaten
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Transformation der 28x28 Bilder als Vektoren der Länge 28*28
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Normalisierung der Daten zwischen 0 und 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Encodieren der Integer-Klassen (0-9) in One-Hot Vektoren
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# relu layers
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# Verwenden von Adam als Optimizer
model.compile(loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(lr=0.001, decay=0.1),
    metrics=['accuracy'])

# Training des Modells
history = model.fit(x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_test, y_test))

def plot_training(training_loss, validation_loss, training_accuracy, validation_accuracy):
    """ Plot von Accuracy und Loss der Trainings- und Validierungs-Daten"""
    plt.figure(1)
    plt.plot(training_accuracy)
    plt.plot(validation_accuracy)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.figure(2)
    plt.plot(training_loss)
    plt.plot(validation_loss)
    plt.title('cross-entropy loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Plot von Accuracy und Loss über das Keras History-Object
plot_training(training_loss=history.history['loss'],
    validation_loss=history.history['val_loss'],
    training_accuracy=history.history['acc'],
    validation_accuracy=history.history['val_acc'])

# Genauigkeit des trainierten Modells
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
