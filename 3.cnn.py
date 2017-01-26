import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 28, 28, 1)  # 1: channel (grayscale)
X_test = X_test.reshape(-1, 28, 28, 1)
y_train = np_utils.to_categorical(y_train, nb_classes=10)
y_test = np_utils.to_categorical(y_test, nb_classes=10)

model = Sequential()

# Conv layer 1 output shape (28, 28, 32)
model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',
    dim_ordering='tf',
    input_shape=(28, 28, 1),  # 1: channel, 28x28: height, width
))
model.add(Activation('relu'))

# Pooling layer 1 (max) output shape (14, 14, 32)
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    border_mode='same',
))

# Conv layer 2 output shape (14, 14, 64)
model.add(Convolution2D(64, 5, 5, border_mode='same'))
model.add(Activation('relu'))

# Pooling layer 2 (max) output shape (7, 7, 64)
model.add(MaxPooling2D(
    pool_size=(2, 2),
    border_mode='same',
))

# Fully connected layer 1 input shape (64*7*7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# optimizer
adam = Adam(lr=1e-4)

model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(X_train, y_train, nb_epoch=1, batch_size=32)

# Testing
loss, accuracy = model.evaluate(X_test, y_test)

print '\nTest loss:', loss
print 'Test accuracy:', accuracy