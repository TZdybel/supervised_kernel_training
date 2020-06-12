import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D
from keras.callbacks import LambdaCallback
from keras import optimizers
from tqdm.keras import TqdmCallback
import numpy as np

def prepare_network_model(num_of_kernels):
    model = Sequential()
    model.add(Conv2D(num_of_kernels, 3, padding='same', input_shape=(3, 3, 1)))
    sgd = optimizers.SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='mean_squared_error')
    model.summary()
    return model

def train(training_samples, samples_labels, num_of_epochs=100, batch_size=1024, validation_split=0.0):
    model = prepare_network_model(samples_labels.shape[-1])
    conv_weights = []

    get_weights = LambdaCallback(on_epoch_end=lambda batch, logs: conv_weights.append(np.array([np.rot90(x, 3) for x in model.layers[0].get_weights()[0].squeeze()[::-1].T])))

    history_temp = model.fit(training_samples, samples_labels,
                            batch_size=batch_size,
                            epochs=num_of_epochs,
                            validation_split=validation_split,
                            callbacks=[get_weights, TqdmCallback(verbose=1)],
                            verbose=0)

    return np.array(conv_weights)
