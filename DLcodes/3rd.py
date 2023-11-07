import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense , Conv2D , Dropout , Flatten , MaxPooling2D
import numpy as np
import matplotlib.pyplot as plt
import random
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data() 
input_shape = (28,28,1)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

x_train = x_train.astype('float32')
x_test =  x_test.astype('float32')

# Scaling data between 0 to 1 
x_train = x_train / 255
x_test = x_test / 255

print("Shape of traning :", x_train.shape)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(100, activation="relu"),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer='adam', 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

model.summary()
model.fit(x_train, y_train, epochs=3, batch_size = 32)

test_loss , test_acc = model.evaluate(x_test,y_test)
print("Loss%.3f" %test_loss)
print("Accuracy%.3f" %test_acc)

# Showing img at position[] from dataset
predictions = np.argmax(model.predict(X_test), axis=-1)
accuracy_score(y_test, predictions)
n=random.randint(0,9999)
plt.imshow(X_test[n],)
plt.show()
predicted_value=model.predict(X_test)
print("Handwritten number in the image is= %d" %np.argmax(predicted_value[n]))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0]) #Test loss: 0.0296396646054
print('Test accuracy:', score[1])

