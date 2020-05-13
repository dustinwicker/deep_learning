from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt

# Load in mnist data set and perform train/test split
(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_dim = 784 # 28*28
output_dim = nb_classes = 10
batch_size = 128
nb_epoch = 20

# Reshape x_train and x_test from two-dimensional data into one-dimensional data
x_train = x_train.reshape(60000, input_dim)
x_test = x_test.reshape(10000, input_dim)

# Change type from uint8 to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize each vector by dividing each element by 255 (this is the maximum value of the RGB color scale)
x_train /= 255
x_test /= 255

# One hot encode target variable
y_train = to_categorical(y_train, num_classes=nb_classes)
y_test = to_categorical(y_test, num_classes=nb_classes)

# Plot some of the images
# Set figure size
plt.figure(figsize=(20,5))

# Select subplot to use
plt.subplot(141)
# Use imshow to display as an image
plt.imshow(x_train[123].reshape(28,28), cmap='gray')
plt.title(f"Label of the image {y_train[123]}")

# Select subplot to use
plt.subplot(142)
# Use imshow to display as an image
plt.imshow(x_train[124].reshape(28,28), cmap='gray')
plt.title(f"Label of the image {y_train[124]}")

# Select subplot to use
plt.subplot(143)
# Use imshow to display as an image
plt.imshow(x_train[125].reshape(28,28), cmap='gray')
plt.title(f"Label of the image {y_train[125]}")

# Select subplot to use
plt.subplot(144)
# Use imshow to display as an image
plt.imshow(x_train[126].reshape(28,28), cmap='gray')
plt.title(f"Label of the image {y_train[126]}")

# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(1028, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(1028, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Obtain structure of our artificial neural network (ANN) model
model.summary()

# Compile the model
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

### Assignment (Lesson 3)
# 1
# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(32, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(16, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Obtain structure of our artificial neural network (ANN) model
model.summary()

# Compile the model
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

# 2
# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(1024, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(512, activation="relu"))
# Add third dense layer
model.add(Dense(256, activation="relu"))
# Add fourth dense layer
model.add(Dense(128, activation="relu"))
# Add fifth dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Obtain structure of our artificial neural network (ANN) model
model.summary()

# Compile the model
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

### Assignment (Lesson 4)
# 1
# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="tanh"))
# Add second dense layer
model.add(Dense(64, activation="tanh"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")


# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="sigmoid"))
# Add second dense layer
model.add(Dense(64, activation="sigmoid"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")


model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

# 2
# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="tanh"))
# Add second dense layer
model.add(Dense(64, activation="tanh"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")


# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="sigmoid"))
# Add second dense layer
model.add(Dense(64, activation="sigmoid"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")


model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

### Assignment (Lesson 5)
# 1
# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=8, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=128, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Compile the model
model.compile(optimizer="sgd", loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=784, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

# 2
# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Create an optimizer with the desired properties
opt = SGD(learning_rate=0.01)
# Compile the model
model.compile(optimizer=opt, loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=128, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Create an optimizer with the desired properties
opt = SGD(learning_rate=100)
# Compile the model
model.compile(optimizer=opt, loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=128, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Build our Artifical Neural Network (ANN) model
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(128, input_shape=(784,), activation="relu"))
# Add second dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(10, activation="softmax"))

# Create an optimizer with the desired properties
opt = SGD(learning_rate=0.0000001)
# Compile the model
model.compile(optimizer=opt, loss='categorical_hinge', metrics=['accuracy'])

# Train the model
model.fit(x=x_train, y=y_train, batch_size=128, epochs=nb_epoch, verbose=1)

# Evaluate the model using the test set
score = model.evaluate(x=x_test, y=y_test, verbose=1)
print(f"Test score: {score[0]}")
print(f"Test accuracy: {score[1]}")