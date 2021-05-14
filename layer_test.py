from tensorflow.keras.models import Sequential, clone_model, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np

image_input = Input((90, 80, 4))
image_network = Conv2D(filters = 16,kernel_size = (8,8),strides = 4,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2))(image_input)
image_network = Conv2D(filters = 32,kernel_size = (4,4),strides = 2,data_format="channels_last", activation = 'relu',kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2))(image_network)

image_network = Flatten()(image_network)
image_network = Dense(256,activation = 'relu', kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2))(image_network)

image_model = Model(inputs=image_input, outputs=image_network)

print(image_model.summary())

action_input = Input((1))
action_network = Dense(32, activation="relu")(action_input)
#action_network = action_input

action_model = Model(inputs=action_input, outputs=action_network)

print(action_model.summary())

print(action_model.output.shape)
print(image_model.output.shape)
print(np.shape(action_model.output))
print(np.shape(image_model.output))

combined = concatenate([image_model.output, action_model.output])

possible_actions = [0, 1, 2, 3]
action_selection = Dense(len(possible_actions), activation = 'linear')(combined)

model = Model(inputs = [image_model.input, action_model.input], outputs = action_selection)
print(model.summary())
print(model.inputs)

#print(combined)
