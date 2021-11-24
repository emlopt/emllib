Tutorial
========

To make more clear how Empirical Model Learning can be used, here we present a toy example.
We assume to have an artificial dataset such that:

.. math::
    X = (x_0, x_1) \\
    f(x_0, x_1) = (x_0 \cdot x_1)^2

Which can be generated as follows:

    .. code-block:: python

        import numpy as np

        ninput = 2
        nsamples = 3000
        nsamples_test = 1000

        X = np.random.rand(nsamples, ninput)
        y = (X[:,0] * X[:,1])**2
        # creating dataset for test
        Xt = np.random.rand(nsamples_test, ninput)
        yt = (Xt[:,0] * Xt[:,1])**2

Assuming that the form of the function :math:`f` \is unknown, we want to solve the following problem:

.. math::
    maximize_{x_0, x_1, y}~~x_0 + x_1 + f(x_0, x_1) \\
    x_0 \leq 0.2 \cdot x_1
    \\
    x_0 + x_1 + f(x_0, x_1) \leq 1 \\

The elements we are going to need to solve the problem are:

* A Deep Neural network approximating the function :math:`f`

* A formulaiton of the combinatorial problem

* A way to embed the model into the combinatorial problem (provided by EML)


Build the Neural Network
------------------------

The structure of the neural network depends on the EML library itself.
In order to use the model we need to encode it into the combinatorial problem,
this requires having an ad hoc parser relative to the python library used to build the neural network.
At the moment being EML supports only sequential :py:mod:`keras` models.
\\
A general formulation of the neural network might look like this:

    .. code-block:: python

        from tensorflow.keras import backend as K
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense

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
                model.add(Dense(1, activation='linear'))
                return model

        width = 8
        hlayers = 4
        actfun = 'linear'

        # create model
        model = SimpleModelBuilder(ninput, actfun, hlayers, width)
        model = model.build_model()

After defining the model we can train it using the data sample previously generated:

    .. code-block:: python

        from tensorflow.keras.callbacks import EarlyStopping

        batch_size = 256
        epochs = 200
        seed = 42

        model.compile(loss='mse', optimizer='adam')

        # train model
        callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
        model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, verbose=1, callbacks=callbacks)

        # test model
        eval = model.evaluate(x=Xt, y=yt, batch_size=batch_size, verbose=1)
        print(eval)

Once we have trained the model we are going to store it for later usage:

    .. code-block:: python

        from tensorflow.keras.models import model_from_json

        # saving model
        model.save_weights('nn_reg.h5')
        with open('nn_reg.json', 'w') as f:
            f.write(model.to_json())

        def laod_keras_net():
            with open('nn_reg.json') as f:
                knet = model = model_from_json(f.read())
            wgt_fname = os.path.join('nn_reg.h5')
            knet.load_weights(wgt_fname)
            return knet

What we need now \is a way to translate the :py:mod:`keras` network into a format readable by the EML framework. This function \is provided by the function :py:mod:`keras_reader.read_keras_sequential`:

    .. code-block:: python

        from eml.net.reader import keras_reader

        def convert_keras_net(knet):
            net = keras_reader.read_keras_sequential(knet)
            return net

Problem Formulation \& Neural Network Embedding
-----------------------------------------------

The combinatorial problem can be defined using either :py:mod:`ortools` \or :py:mod:`cplex`, which are supported by the EML library.
Future improvement of this package will include other frameworks.\
The use of one of the two libraries implies different syntaxes to define the problem.

OR\-Tools Backend
`````````````````

First, we need to create an instance of the optimization model:

    .. code-block:: python

        import eml.backend.ortool_backend as ortools_backend

        bkd = ortools_backend.OrtoolsBackend()
        mdl = bkd.new_model()

Then we need to load the neural network previously trained:

    .. code-block:: python

        knet = laod_keras_net()
        net = convert_keras_net(knet)

To encode the Deep Learning model we need to instantiate the variable representing respectively the \input and the output of the neural network:

    .. code-block:: python

        X0_var = mdl.NumVar(lb=0, ub=1, name='in_var0')
        X1_var = mdl.NumVar(lb=0, ub=1, name='in_var1')
        Y_var = mdl.NumVar(lb=0, ub=1, name='out_var')

Finally, we can encode the network:

    .. code-block:: python

        from eml.net.embed import encode

        encode(bkd, net, mdl, [X0_var, X1_var], Y_var, 'net_econding')

To conclude we can specify additional variable \and constraint, and set the objective function:

    .. code-block:: python

        R_var = bkd.xpr_sum(mdl, [X0_var, X1_var, Y_var])
        mdl.Add(X0_var <= 0.2 * X1_var)
        mdl.Add(R_var <= 1)

        bkd.set_obj(mdl, 'max', R_var)


CPLEX Backend
`````````````

First, we need to create an instance of the optimization model:

    .. code-block:: python

        bkd = cplex_backend.CplexBackend()
        mdl = cpx.Model()

Then we need to load the neural network previously trained:

    .. code-block:: python

        knet = laod_keras_net()
        net = convert_keras_net(knet)

To encode the Deep Learning model we need to instantiate the variable representing respectively the \input and the output of the neural network:

    .. code-block:: python

        X0_var = mdl.continuous_var(lb=0, ub=1, name='in_var0')
        X1_var = mdl.continuous_var(lb=0, ub=1, name='in_var1')
        Y_var = mdl.continuous_var(lb=0, ub=1, name='out_var')

Finally, we can encode the network:

    .. code-block:: python

        encode(bkd, net, mdl, [X0_var, X1_var], Y_var, 'net_econding')

To conclude we can specify additional variable \and constraint, and set the objective function:

    .. code-block:: python

        R_var = mdl.sum([X0_var, X1_var, Y_var])
        mdl.add_constraint(X0_var <= 0.2 * X1_var)
        mdl.add_constraint(R_var <= 1)
        mdl.set_objective('max', R_var)

Solve the problem
-----------------

The final step, which produces the solution to our problem, requires just to invoke the function :py:mod:`solve` on the combinatorial model:

    .. code-block:: python

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

