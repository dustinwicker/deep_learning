import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Increase maximum width in characters of columns - will put all columns in same line in console readout
pd.set_option('expand_frame_repr', False)
# Be able to read entire value in each column (no longer truncating values)
pd.set_option('display.max_colwidth', -1)
# Increase number of rows printed out in console
pd.set_option('display.max_rows', 200)

# Load in mnist data set and perform train/test split
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

input_dim = 28*28 # 784 x_train.shape[1] * x_train.shape[2]
output_dim = nb_classes = len(np.unique(y_train)) # 10

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

# Create dict of numerical target to actual category
target_labels = {0: "tshirt/top", 1: "trouser", 2: "pullover", 3: "dress", 4: "coat", 5: "sandal", 6: "shirt",
                 7: "sneaker", 8: "bag", 9: "ankle_boot"}

# Plot some of the images
# Set figure size
plt.figure(figsize=(20,7))

# Select subplot to use
plt.subplot(141)
# Choose example from train
train_example = 4
# Use imshow to display as an image
plt.imshow(x_train[train_example].reshape(28,28), cmap='gray')
plt.title(f"Label of the image: {target_labels[np.where(y_train[train_example] == 1)[0][0]]}")

# Select subplot to use
plt.subplot(142)
# Choose example from train
train_example = 14
# Use imshow to display as an image
plt.imshow(x_train[train_example].reshape(28,28), cmap='gray')
plt.title(f"Label of the image: {target_labels[np.where(y_train[train_example] == 1)[0][0]]}")

# Select subplot to use
plt.subplot(143)
# Choose example from train
train_example = 44
# Use imshow to display as an image
plt.imshow(x_train[train_example].reshape(28,28), cmap='gray')
plt.title(f"Label of the image: {target_labels[np.where(y_train[train_example] == 1)[0][0]]}")

# Select subplot to use
plt.subplot(144)
# Choose example from train
train_example = 24
# Use imshow to display as an image
plt.imshow(x_train[train_example].reshape(28,28), cmap='gray')
plt.title(f"Label of the image: {target_labels[np.where(y_train[train_example] == 1)[0][0]]}")

# Build our Artifical Neural Network (ANN) models
# Define parameters
nb_epoch = 20

# Create empty DataFrames to append results to
nn_model_train_results_one = pd.DataFrame()
nn_model_test_results_one = pd.DataFrame()
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(1028, input_shape=(input_dim,), activation="relu"))
# Add second dense layer
model.add(Dense(1028, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(output_dim, activation="softmax"))

# Compile the model
for loss_metric in ["categorical_crossentropy", "categorical_hinge"]:
    print("-"*40)
    print(loss_metric)
    model.compile(optimizer="sgd", loss=loss_metric, metrics=['accuracy'])

    # Train the model
    for batch in [8, 16, 32, 64, 128]:
        print(batch)
        model.fit(x=x_train, y=y_train, batch_size=batch, epochs=nb_epoch, verbose=2)
        # Append each training around accuracy
        nn_model_train_results_one[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = model.history.history['accuracy']

        # Evaluate the model using the test set
        score = model.evaluate(x=x_test, y=y_test, verbose=1)
        print(f"Test score: {score[0]}")
        print(f"Test accuracy: {score[1]}")
        # Append training accuracy of each model
        nn_model_test_results_one[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = [score[1]]


# Create empty DataFrames to append results to
nn_model_train_results_two = pd.DataFrame()
nn_model_test_results_two = pd.DataFrame()
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(1028, input_shape=(input_dim,), activation="tanh"))
# Add second dense layer
model.add(Dense(1028, activation="tanh"))
# Add last layer, i.e. the output layer
model.add(Dense(output_dim, activation="softmax"))

# Compile the model
for loss_metric in ["categorical_crossentropy", "categorical_hinge"]:
    print("-"*40)
    print(loss_metric)
    model.compile(optimizer="sgd", loss=loss_metric, metrics=['accuracy'])

    # Train the model
    for batch in [8, 16, 32, 64, 128]:
        print(batch)
        model.fit(x=x_train, y=y_train, batch_size=batch, epochs=nb_epoch, verbose=2)
        # Append each training around accuracy
        nn_model_train_results_two[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = model.history.history['accuracy']

        # Evaluate the model using the test set
        score = model.evaluate(x=x_test, y=y_test, verbose=1)
        print(f"Test score: {score[0]}")
        print(f"Test accuracy: {score[1]}")
        # Append training accuracy of each model
        nn_model_test_results_two[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = [score[1]]


# Create empty DataFrames to append results to
nn_model_train_results_three = pd.DataFrame()
nn_model_test_results_three = pd.DataFrame()
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(1028, input_shape=(input_dim,), activation="sigmoid"))
# Add second dense layer
model.add(Dense(1028, activation="sigmoid"))
# Add last layer, i.e. the output layer
model.add(Dense(output_dim, activation="softmax"))

# Compile the model
for loss_metric in ["categorical_crossentropy", "categorical_hinge"]:
    print("-"*40)
    print(loss_metric)
    model.compile(optimizer="sgd", loss=loss_metric, metrics=['accuracy'])

    # Train the model
    for batch in [8, 16, 32, 64, 128]:
        print(batch)
        model.fit(x=x_train, y=y_train, batch_size=batch, epochs=nb_epoch, verbose=2)
        # Append each training around accuracy
        nn_model_train_results_three[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = model.history.history['accuracy']

        # Evaluate the model using the test set
        score = model.evaluate(x=x_test, y=y_test, verbose=1)
        print(f"Test score: {score[0]}")
        print(f"Test accuracy: {score[1]}")
        # Append training accuracy of each model
        nn_model_test_results_three[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = [score[1]]


# Create empty DataFrames to append results to
nn_model_train_results_four = pd.DataFrame()
nn_model_test_results_four = pd.DataFrame()
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(1024, input_shape=(input_dim,), activation="relu"))
# Add second dense layer
model.add(Dense(512, activation="relu"))
# Add third dense layer
model.add(Dense(256, activation="relu"))
# Add fourth dense layer
model.add(Dense(128, activation="relu"))
# Add fifth dense layer
model.add(Dense(64, activation="relu"))
# Add last layer, i.e. the output layer
model.add(Dense(output_dim, activation="softmax"))

# Compile the model
for loss_metric in ["categorical_crossentropy", "categorical_hinge"]:
    print("-"*40)
    print(loss_metric)
    model.compile(optimizer="sgd", loss=loss_metric, metrics=['accuracy'])

    # Train the model
    for batch in [8, 16, 32, 64, 128]:
        print(batch)
        model.fit(x=x_train, y=y_train, batch_size=batch, epochs=nb_epoch, verbose=2)
        # Append each training around accuracy
        nn_model_train_results_four[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = model.history.history['accuracy']

        # Evaluate the model using the test set
        score = model.evaluate(x=x_test, y=y_test, verbose=1)
        print(f"Test score: {score[0]}")
        print(f"Test accuracy: {score[1]}")
        # Append training accuracy of each model
        nn_model_test_results_four[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = [score[1]]

# Create empty DataFrames to append results to
nn_model_train_results_five = pd.DataFrame()
nn_model_test_results_five = pd.DataFrame()
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(1024, input_shape=(input_dim,), activation="tanh"))
# Add second dense layer
model.add(Dense(512, activation="tanh"))
# Add third dense layer
model.add(Dense(256, activation="tanh"))
# Add fourth dense layer
model.add(Dense(128, activation="tanh"))
# Add fifth dense layer
model.add(Dense(64, activation="tanh"))
# Add last layer, i.e. the output layer
model.add(Dense(output_dim, activation="softmax"))

# Compile the model
for loss_metric in ["categorical_crossentropy", "categorical_hinge"]:
    print("-"*40)
    print(loss_metric)
    model.compile(optimizer="sgd", loss=loss_metric, metrics=['accuracy'])

    # Train the model
    for batch in [8, 16, 32, 64, 128]:
        print(batch)
        model.fit(x=x_train, y=y_train, batch_size=batch, epochs=nb_epoch, verbose=2)
        # Append each training around accuracy
        nn_model_train_results_five[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = model.history.history['accuracy']

        # Evaluate the model using the test set
        score = model.evaluate(x=x_test, y=y_test, verbose=1)
        print(f"Test score: {score[0]}")
        print(f"Test accuracy: {score[1]}")
        # Append training accuracy of each model
        nn_model_test_results_five[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = [score[1]]

# Create empty DataFrames to append results to
nn_model_train_results_six = pd.DataFrame()
nn_model_test_results_six = pd.DataFrame()
# Create the model object
model = Sequential()
# Add first dense layer
model.add(Dense(1024, input_shape=(input_dim,), activation="sigmoid"))
# Add second dense layer
model.add(Dense(512, activation="sigmoid"))
# Add third dense layer
model.add(Dense(256, activation="sigmoid"))
# Add fourth dense layer
model.add(Dense(128, activation="sigmoid"))
# Add fifth dense layer
model.add(Dense(64, activation="sigmoid"))
# Add last layer, i.e. the output layer
model.add(Dense(output_dim, activation="softmax"))

# Compile the model
for loss_metric in ["categorical_crossentropy", "categorical_hinge"]:
    print("-"*40)
    print(loss_metric)
    model.compile(optimizer="sgd", loss=loss_metric, metrics=['accuracy'])

    # Train the model
    for batch in [8, 16, 32, 64, 128]:
        print(batch)
        model.fit(x=x_train, y=y_train, batch_size=batch, epochs=nb_epoch, verbose=2)
        # Append each training around accuracy
        nn_model_train_results_six[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = model.history.history['accuracy']

        # Evaluate the model using the test set
        score = model.evaluate(x=x_test, y=y_test, verbose=1)
        print(f"Test score: {score[0]}")
        print(f"Test accuracy: {score[1]}")
        # Append training accuracy of each model
        nn_model_test_results_six[model.get_config()['layers'][0]['config']['activation'] + '_' + str(batch) + '_' +
                                   model.loss] = [score[1]]
