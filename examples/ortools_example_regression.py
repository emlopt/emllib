
import os
import numpy as np

from keras import backend as K
from keras import Model
from keras.layers import Input, Dense
from keras.models import model_from_json

import eml.backend.ortool_backend as ortools_backend
from eml.net.embed import encode
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
        # Build a simple neural model
        inp = Input(shape=(self.ninput,))
        tmp = inp
        for i in range(self.hlayers):
            tmp = Dense(self.width, activation='relu')(tmp)
        out = Dense(1, activation=self.actfun)(tmp)
        model = Model(inp, out)
        # Return the model
        return model

ninput = 2
nsamples = 3000
nsamples_test = 1000
width = 8
hlayers = 4
actfun = K.relu
batch_size = 256
epochs = 200
seed = 42

np.random.seed(seed)

# creating dataset for training
X = np.random.rand(nsamples, ninput)
y = np.random.rand(nsamples)
# creating dataset for test
Xt = np.random.rand(nsamples_test, ninput)
yt = np.random.rand(nsamples_test)

# create model
model = SimpleModelBuilder(ninput, actfun, hlayers, width)
model = model.build_model()
model.compile(loss='mse', optimizer='adam')


# train model
model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, verbose=1)

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

bkd = ortools_backend.OrtoolsBackend()
mdl = bkd.new_model()

knet = laod_keras_net()
net = convert_keras_net(knet)

# create variables
X0_var = mdl.NumVar(lb=0, ub=1, name='in_var0')
X1_var = mdl.NumVar(lb=0, ub=1, name='in_var1')
Y_var = mdl.NumVar(lb=0, ub=1, name='out_var')

# encode model
encode(bkd, net, mdl, [X0_var, X1_var], Y_var, 'net_econding')

# create constraints
mdl.Add(X0_var <= 0.2 * X1_var)

# build objective var
R_var = bkd.xpr_sum(mdl, [X0_var, X1_var, Y_var])
bkd.set_obj(mdl, 'max', R_var)

print('=== Starting the solution process')
sol = bkd.solve(mdl, 30)

if sol is None:
    print('=== NO SOLUTION FOUND')
else:
    print('=== SOLUTION DATA')
    print('Solution time: {:.3f} (sec)'.format(sol['time']))
    print('Solver status: {}'.format(sol['status']))
    print('X0: {}'.format(X0_var.solution_value()))
    print('X1: {}'.format(X1_var.solution_value()))
    print('Y: {}'.format(Y_var.solution_value()))
    print('Cost: {}'.format(sol['obj']))