#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import keras.models as kmodels
import keras.layers as klayers

from eml.net import describe

# import importlib

# Import custom modules
# path = os.path.join(os.path.dirname(__file__), '..')
# path = os.path.abspath(path)
# if not path in sys.path:
#     sys.path.insert(1, path)
# del path
# import model
# importlib.reload(model)

def read_keras_sequential(kmodel):
    """ Import nueral network model from keras

    Casts neural network into custom representation, available at
    :obj:`eml.net.describe.DTNet`

    Parameters
    ----------
        kmodel : obj:`keras.models.Sequential` 
            Trained keras neural network

    Returns
    -------
        Neural Network : :obj:`eml.net.describe.DNRNet`
            Neural Network with custom representation

    Raises
    ------
        ValueError
            If the layer type is not supported

    """ 
    # Build a DNR network model
    net = describe.DNRNet()
    # Add input layer
    kls = kmodel.layers
    layer = describe.DNRInput(input_shape=kls[0].input_shape[1:])
    net.add(layer)
    # Loop over the layers of the keras network
    for k, klayer in enumerate(kls):
        if klayer.__class__ == klayers.Dense:
            wgt, bias = klayer.get_weights()
            act = klayer.get_config()['activation']
            layer = describe.DNRDense(wgt, bias, act)
            net.add(layer)
        else:
            raise ValueError('Unsupported layer type')
    # Return the network
    return net


# if __name__ == '__main__':
#     # Build a random training set
#     ns, na, nh, no = 100, 3, 2, 1
#     datax = np.random.random((ns, na))
#     datay = np.random.random((ns, no))
#     # Learn a super-simple keras model
#     mdl = kmodels.Sequential()
#     mdl.add(klayers.Dense(nh, input_shape=(na,), activation='relu'))
#     mdl.add(klayers.Dense(no))
#     mdl.compile(optimizer='rmsprop', loss='mse')
#     mdl.fit(datax, datay, epochs=10)

#     # Convert the network in DNR format
#     net = read_keras_sequential(mdl)
#     print(net)

#     # Obtain keras predictions
#     kx = np.random.random((10, na))
#     ky = mdl.predict(kx)
#     # Obtain DNR net predictions
#     for i in range(kx.shape[0]):
#         dnr_eval = net.evaluate(kx[i])
#         diffs = np.abs(dnr_eval.layer(-1) - ky[i])
#         assert(np.all(diffs < 1e-4))


