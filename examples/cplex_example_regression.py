
import os
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping

import docplex.mp.model as cpx

from eml.net.embed import encode
from eml.backend import cplex_backend
from eml.net.reader import keras_reader


class SimpleModelBuilder(object):
    """Build an ML model for this task"""
    def __init__(self, ninput, actfun, hlayers, width):
        assert(width >= 1)
        assert(hlayers >= 0)
        assert(ninput >= 1)
        super(SimpleModelBuilder, self).__init__()
        self.ninput = ninput
        self.actfun = actfun
        self.hlayers = hlayers
        self.width = width

    def build_model(self):
        model = Sequential()
        model.add(Dense(1, activation='relu', input_shape=(self.ninput,)))
        for i in range(self.hlayers):
            model.add(Dense(self.width, activation='relu'))
        model.add(Dense(1, activation=actfun))
        return model

ninput = 2
nsamples = 3000
nsamples_test = 1000
width = 8
hlayers = 4
actfun = 'linear'
batch_size = 256
epochs = 200
seed = 42

np.random.seed(seed)

# creating dataset for training
X = np.random.rand(nsamples, ninput)
y = (X[:,0] * X[:,1])**2
# creating dataset for test
Xt = np.random.rand(nsamples_test, ninput)
yt = (Xt[:,0] * Xt[:,1])**2

# create model
model = SimpleModelBuilder(ninput, actfun, hlayers, width)
model = model.build_model()
model.compile(loss='mse', optimizer='adam')


# train model
callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks)

# test model
eval = model.evaluate(x=Xt, y=yt, batch_size=batch_size, verbose=1)
print(eval)

# saving model
model.save_weights('nn_reg.h5')
with open('nn_reg.json', 'w') as f:
    f.write(model.to_json())

# build optimization problem

def laod_keras_net():
    with open('nn_reg.json') as f:
        knet = model = model_from_json(f.read())
    wgt_fname = os.path.join('nn_reg.h5')
    knet.load_weights(wgt_fname)
    return knet

def convert_keras_net(knet):
    net = keras_reader.read_keras_sequential(knet)
    return net

bkd = cplex_backend.CplexBackend()
mdl = cpx.Model()

knet = laod_keras_net()
net = convert_keras_net(knet)

# create variables
X0_var = mdl.continuous_var(lb=0, ub=1, name='in_var0')
X1_var = mdl.continuous_var(lb=0, ub=1, name='in_var1')
Y_var = mdl.continuous_var(lb=0, ub=1, name='out_var')

# encode model
encode(bkd, net, mdl, [X0_var, X1_var], Y_var, 'net_econding')

# create constraints
mdl.add_constraint(X0_var <= 0.2 * X1_var)

# build objective var
R_var = mdl.sum([X0_var, X1_var, Y_var])
mdl.set_objective('max', R_var)

# Solve
mdl.set_time_limit(30)
print('=== Starting the solution process')
sol = mdl.solve()

if sol is None:
    print('=== NO SOLUTION FOUND')
else:
    print('=== SOLUTION DATA')
    print('Solution time: {:.2f} (sec)'.format(mdl.solve_details.time))
    print('Solver status: {}'.format(sol.solve_details.status))
    print('X0: {}'.format(sol[X0_var]))
    print('X1: {}'.format(sol[X1_var]))
    print('Y: {}'.format(sol[Y_var]))
    print('Cost: {}'.format(sol[R_var]))
