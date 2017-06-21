import keras
from keras.models import Sequential
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 15

# Laden des Datensatzes und Aufteilung in 60.000 Trainings- und 10.000 Testdaten
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Dimension der Daten erweitern um Kanal
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalisierung der Daten zwischen 0 und 1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Encodieren der Integer-Klassen (0-9) in One-Hot Vektoren
y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

# Convolutional Neural Network
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

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
