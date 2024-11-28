from tensorflow.keras.datasets import mnist
import requests

# Suppress progress bar messages using requests.get instead of urlretrieve
response = requests.get('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')
with open('mnist.npz', 'wb') as f:
  f.write(response.content)

# Load the MNIST dataset from the downloaded file
(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')

# Preprocess data (normalize pixel values)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Reshape data for CNN (add channel dimension)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# One-hot encode labels (modify if using different number of classes)
y_train = tf.keras.utils.to_categorical(y_train, 10) # 10 classes for digits 0-9
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define the CNN model (modify architecture for better performance)
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (adjust epochs and other hyperparameters)
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Make predictions (optional)
predictions = model.predict(x_test)
