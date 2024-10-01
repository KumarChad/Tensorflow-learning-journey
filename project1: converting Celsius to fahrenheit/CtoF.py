import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt

#data set of imputs and outputs
cel_q = np.array([-40, -10, 0 ,8 ,15 ,22 ,38], dtype=float)
far_q = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(cel_q):
    print("{} degrees Celcius = {} degrees Fahrenhet".format(c, far_q[i]))

#creating the model
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

#assembling the layers into the model
model = tf.keras.Sequential([l0])
#compiling model with loss and optimizer function
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))

#training the model
historyk = model.fit(cel_q, far_q, epochs=500, verbose=False)
print("finished training the model")

#display training stats
plt.xlabel("epoch number")
plt.ylabel("loss magnitude")
plt.plot(historyk.history['loss'])
plt.show()

result = model.predict(np.array([100.0]))

print("prediction for 100 degrees cel is {} degrees fahrenheit".format(result[0][0]))

