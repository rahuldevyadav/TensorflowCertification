import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

data = keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = data.load_data()

# index = 0
# np.set_printoptions(linewidth=320)
# print(X_train[index])
# plt.imshow(X_train[index], cmap='gray')
# plt.show()
X_train = X_train/255
X_test = X_test/255

"""
Writing a callback 
"""
class MyCallback(keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('loss')<0.4):
            print("\n Loss is low so canceling training")
            self.model.stop_training = True

callbacks = MyCallback()

model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)),
                          keras.layers.Dense(128, activation = tf.nn.relu),
                          keras.layers.Dense(10, activation = tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(), loss= 'sparse_categorical_crossentropy', metrics = 'accuracy')

model.fit(X_train,y_train,epochs=5, callbacks=callbacks)
model.evaluate(X_test, y_test)
