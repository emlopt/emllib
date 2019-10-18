#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy as np
import itertools

# ============================================================================
# Base classes and functions
# ============================================================================

def act_eval(activation, value):
    """ Evaluates the activation function

    Parameters
    ----------
        activation : :mod:`keras.activations`
            Activation function, either `relu` or `linear`
        value : float 
            Input of the activation function

    Returns
    -------
        Activation Function Result : float
            Value computed by the activation function 

    Raises
    ------
        ValueError
            If activation function is invalid (it is not 
            `relu` or `linear`)

    """
    if activation == 'relu':
        return np.maximum(0, value)
    elif activation == 'linear':
        return value
    else:
        raise ValueError('Invalid activation function "%s"' % activation)


class DNRLayer(object):
    """ Class used to shape neural network layers

    Layer of the network. It is composed by a sequence of 
    neurons with lower and upper bound. Multiple layer together 
    define a neural network.

    Attributes
    ----------
        lb_ : list(float)
            List of lower bounds for this layer
        ub_ : list(float)
            List of upper bounds for this layer 
        idx_ : int
            Index for this layer
        net_ : string
            Name of the neural network 
            where the layer is in 

    Parameters
    ----------
        lb : float
            List of lower bounds for this layer
        ub : float 
            List of upper bounds for this layer 

    """
    def __init__(self, lb, ub):
        super(DNRLayer, self).__init__()
        assert(lb.shape == ub.shape)
        self.lb_ = lb
        self.ub_ = ub
        self.idx_ = 0
        self.net_ = None

    def network(self):
        """ Get network name

        Returns
        -------
            Name : string
                Name of the network

        """
        return self.net_

    def size(self):
        """ Get size layer

        Returns
        -------
            Size of the layer : int
                Number of neuron in the layer

        """
        return len(self.lb_)
    
    def neuron(self, *idx):
        """ Get neuron specific neuron

        Every neuron in a network can be identified 
        by a index that specifies the layer and the
        position in the layer of it.

        Parameters
        ----------
            idx : (int, int)
                index

        Returns
        -------
            Neuron : :obj:`eml.net.describe.DNRNeuron`
                Neuron fuond at the index given  

        """
        if len(idx) == 1:
            idx = idx[0]
        return DNRNeuron(self, idx)

    def neurons(self):
        """ Get all the neurons in this layer

        Yields
        ------
            Neuron : :obj:`eml.net.describe.DNRNeuron`
                Next neuron in the layer

        """
        it = np.nditer(self.lb_, flags=['multi_index'])
        while not it.finished:
            yield DNRNeuron(self, it.multi_index)
            it.iternext()

    def lb(self):
        """ Get the lower bound of the current layer 

        Returns
        -------
            List of lower bounds: list(float)
                List of the lower bound of the current layer

        """
        return self.lb_[:]

    def update_lb(self, value, tol=1e-4): 
        """ Update lower bounds of the current layer

        Parameters
        ----------
            value : list(float)
                New values for the lower bounds 
            tol : float
                Tollerance

        Returns
        -------
            Acknoledge : bool
                True if executed succesfully, Flase otherwise

        """
        mask = (self.lb_  < value - tol)
        self.lb_[mask] = value
        return mask

    def ub(self):
        """ Get the upper bound of the current layer 

        Returns
        -------
            List of upper bounds: list(float)
                List of the upper bound of the current layer

        """
        return self.ub_[:]

    def update_ub(self, value, tol=1e-4):
        """ Update upper bounds of the current layer

        Parameters
        ----------
            value : list(float)
                New values for the upper bounds 
            tol : float
                Tollerance

        Returns
        -------
            Acknoledge : bool
                True if executed succesfully, Flase otherwise

        """
        mask = (self.ub_ > value+tol)
        self.ub_[mask] = value
        return mask

    def reset_bounds(self):
        """ Reset lower and upper bounds of the current layer

        Raises
        ------
            NotImplementedError
                If not implemented in the instanca

        """
        raise NotImplementedError('This method should be \
            implemented in subclasses')
    def connected(self, *idx):
        """ Get the neurons connected to a specific one in the current layer

        Parameters
        ----------
            idx : (int, int)
                Coordinates of a specific neuron

        Raises
        ------
            NotImplementedError
                If not implemented in the instanca


        """
        raise NotImplementedError('This method should be \
            implemented in subclasses')

    def ltype(self):
        """ Get the type of the layer

        Raises
        ------
            NotImplementedError
                If not implemented in the instanca


        """
        raise NotImplementedError('This method should be \
            implemented in subclasses')

    def __repr__(self):
        """ Representation of the layer

        Returns:
            Representation of the Layer : string 
                String representing the current layer

        """
        return ' '.join('%s' % str(n) for n in self.neurons())


class DNRNeuron(object):
    """ Class used to shape neural network's neurons

    Layers are build using neurons. Eache neuron has a lower and 
    upper bound. Bound represent the range of value the neuron 
    can have.

    Attributes
    ----------
        layer_ : :obj:`eml.net.describe.DNRLayer`
            Layer where the neuron is located
        idx_ : int
            Identifier of the neuron 

    Parameters
    ----------
        layer : :obj:`eml.net.describe.DNRLayer`
            Layer where the neuron is located
        idx : int
            Identifier of the neuron 
    
    """
    def __init__(self, layer, idx):
        super(DNRNeuron, self).__init__()
        self.layer_ = layer
        self.idx_ = idx

    def layer(self):
        """ Get the layer where the neuron is located
        
        Returns
        -------
            Layer : :obj:`eml.net.describe.DNRLayer`
                Layer where the neuron is located

        """
        return self.layer_

    def network(self):
        """ Get the name of the network where the neuron is 

        Returns
        -------
            Name : string
                Name of the network
            
        """
        return self.layer_.network()

    def idx(self):
        """ Get idenfier of the neuron

        Returns
        -------
            Index : (int, int)
                Coordinates of layer and neuron 
                in the netwokr

        """
        return self.layer_.idx_ + self.idx_

    def lb(self):
        """ Get the lower bound of the current neuron

        Returns
        -------
            Lower bound : float
                Lower bound of the current neuron
        
        """
        return self.layer_.lb_[self.idx_]

    def update_lb(self, value, tol=1e-4):
        """ Updates the lower bound of the current neuron
        
        Parameters
        ----------
            value : float
                New lower bound
            tol : float
                Tollerance

        Returns
        -------
            Acknowledge : bool
                True is executed succesfully, 
                False otherwise

        """
        if self.layer_.lb_[self.idx_] < value-tol:
            self.layer_.lb_[self.idx_] = value
            return True
        return False

    def ub(self):
        """ Get the upper bound of the current neuron

        Returns
        -------
            Upper bound : float
                Upper bound of the current neuron
        
        """
        return self.layer_.ub_[self.idx_]

    def update_ub(self, value, tol=1e-4):
        """ Updates the upper bound of the current neuron
        
        Parameters
        ----------
            value : float
                New upper bound
            tol : float
                Tollerance

        Returns
        -------
            Acknowledge : bool
                True is executed succesfully, 
                False otherwise

        """
        if self.layer_.ub_[self.idx_] > value+tol:
            self.layer_.ub_[self.idx_] = value
            return True
        return False

    def connected(self):
        """ Get the neurons connected to the current one

        Returns
        -------
           List of nuerons : list(:obj:`eml.net.describe.DNRNeuron`)

        """
        return self.layer_.connected(self.idx_)

    def __repr__(self):
        """ Representation of the neuron

        Returns
        -------
            Representation of the Neuron : string
                String representing the neuron

        """
        return '%s:[%.3f, %.3f]' % (self.idx(), self.lb(), self.ub())


class DNRActLayer(DNRLayer):
    """ Class used to shape neural network's activation layer

    Therefore layers caracterized by an activation function.

    Attributes
    ----------
        ylb_ : list(float)
            List of lower bounds for the current layer
        yub_ : list(float)
            List of upper bounds for the current layer

    Parameters
    ----------
            lb : list(float) 
                List of lower bounds for the current layer
            ub: list(float)
                List of upper bounds for the current layer 
            ylb: list(float)
                List of lower bounds for the current layer
            yub: list(float)
                List of upper bounds for the current layer 

    """
    def __init__(self, lb, ub, ylb, yub):
        assert(lb.shape == ub.shape)
        assert(ylb.shape == yub.shape)
        super(DNRActLayer, self).__init__(lb, ub)
        self.ylb_ = ylb
        self.yub_ = yub

    def neuron(self, *idx):
        """ Get specific neuron in the current layer

        Parameters
        ----------
            idx : (int, int)    
                Index layer, neuron

        Returns
        -------
            Neuron : :obj:`eml.net.describe.DNRActNeuron`

        """
        if len(idx) == 1:
            idx = idx[0]
        return DNRActNeuron(self, idx)

    def neurons(self):
        """ Get the neurons in the current activaiton layer

        Returns
        -------
            List of Neuron : list(:obj:`eml.net.describe.DNRLayer`)
                List of the neurons in the current layer

        """
        it = np.nditer(self.lb_, flags=['multi_index'])
        while not it.finished:
            yield DNRActNeuron(self, it.multi_index)
            it.iternext()

    def ylb(self):
        """ Get the output lower bound of the layer

        Returns
        -------
            List of lower bounds : list(float)
                List of output lower bounds for the current activation layer
        
        """
        return self.ylb_[:]

    def update_ylb(self, value, tol=1e-4):
        """ Updates output lower bounds of the layer
        
        Parameters
        ----------
            value : list(float)
                List of new lower bounds
            tol : float
                Tollerance

        Returns
        -------
            Acknowledge : bool  
                True if executed succesfully,  
                False otherwise
        """
        mask = (self.ylb_  < value-tol)
        self.ylb_[mask] = value
        return mask

    def yub(self):
        """ Get the output upper bound of the layer

        Returns
        -------
            List of upper bounds : list(float)
                List of output upper bounds for the current activation layer
        
        """
        return self.yub_[:]

    def update_yub(self, value, tol=1e-4):
        """ Updates output upper bounds of the layer
        
        Parameters
        ----------
            value : list(float)
                List of new upper bounds
            tol : float
                Tollerance

        Returns
        -------
            Acknowledge : bool  
                True if executed succesfully,  
                False otherwise
        """
        mask = (self.yub_ > value+tol)
        self.yub_[mask] = value
        return mask

    def reset_bounds(self):
        """ Reset lower and upper bounds of the current layer

        Raises
        ------
            NotImplementedError
                If not implemented in the instance

        """
        raise NotImplementedError('This method should be \
            implemented in subclasses')

    def weights(self, *idx):
        """ Get the weights associated to the neurons in the current layer

        Raises
        -----
            NotImplementedError
                If not implemented in the instance

        """
        raise NotImplementedError('This method should be \
            implemented in subclasses')

    def bias(self, *idx):
        """ Get the bias associated to the neurons in the current layer

        Raises
        -----
            NotImplementedError
                If not implemented in the instance

        """
        raise NotImplementedError('This method should be \
            implemented in subclasses')

    def evaluate(self, data):
        """ Evalueates the output of the layer given a input

        Raises
        -----
            NotImplementedError
                If not implemented in the instance
        
        """
        raise NotImplementedError('This method should be \
            implemented in subclasses')


class DNRActNeuron(DNRNeuron):
    """ Class used to shape neural network's activation neurons 

    Therefore neurons with an activation function 

    Attributes
    ----------
        layer_ : :obj:`eml.net.describe.DNRLayer`
            Layer where the neuron is located
        idx_ : int 
            Index of the neuron 

    Parameters
    ----------
            layer : obj:`eml.net.describe.DNRLayer`
                Layer where the neuron is located
            idx : int 
                Index of the neuron
    
    """
    def __init__(self, layer, idx):
        super(DNRActNeuron, self).__init__(layer, idx)

    def ylb(self):
        """ Get the output lower bound of the neuron

        Returns
        -------
            Lower bound : float
                Output lower bounds for the current activation neuron
        
        """
        return self.layer_.ylb_[self.idx_]

    def update_ylb(self, value, tol=1e-4):
        """ Updates output lower bound of the neuron
        
        Parameters
        ----------
            value : float
                Lower bound
            tol : float
                Tollerance

        Returns
        -------
            Acknowledge : bool  
                True if executed succesfully,  
                False otherwise
        """
        if self.layer_.ylb_[self.idx_] < value-tol:
            self.layer_.ylb_[self.idx_] = value
            return True
        return False

    def yub(self):
        """ Get the output upper bound of the neuron

        Returns
        -------
            Upper bound : float
                Output upper bounds for the current activation neuron
        
        """
        return self.layer_.yub_[self.idx_]

    def update_yub(self, value, tol=1e-4):
        """ Updates output upper bound of the neuron
        
        Parameters
        ----------
            value : float
                Upper bound
            tol : float
                Tollerance

        Returns
        -------
            Acknowledge : bool  
                True if executed succesfully,  
                False otherwise
        """
        if self.layer_.yub_[self.idx_] > value+tol:
            self.layer_.yub_[self.idx_] = value
            return True
        return False

    def weight(self):
        """ Get the weight associated to the neuron 

        Returns
        -------
            Weight : float
                Value corresponding to the weight of the 
                current neuron

        """
        return self.layer_.weights(self.idx_)

    def bias(self):
        """ Get the bias associated to the neurons 

        Returns
        -------
            Bias : float
                Value corresponding to the bias of the 
                current neuron

        """
        return self.layer_.bias(self.idx_)

    def activation(self):
        """ Get the activation function of the neuron

        Returns
        -------
            Activation function : :obj:`keras.activations`
                activation function of the neuron

        """
        return self.layer_.activation_

    def __repr__(self):
        """ Representation of the neuron

        Returns
        -------
            Representation : string
                String representing the neuron

        """
        return '%s:[%.3f, %.3f]/[%.3f, %.3f]' % (self.idx(),
                self.ylb(), self.yub(), self.lb(), self.ub())

# ============================================================================
# Input layer
# ============================================================================

class DNRInput(DNRLayer):
    """ Class used to shape the input layer of the neural network

    Attributes
    ----------
        lb_ : list(float)
            List of lower bounds for the input layer
        ub_ : list(float) 
            list of upper bounds for the input layer
        idx_ : int
            Index of the input layer
        net_ : string
            name of the neural network

    Parameters
    ----------
        input_shape : (int, int)
            Dimensions fo the input (default None)
        lb : list(float)
            Lower bounds for the input layer
        ub : list(float)
            Upper bounds for the input layer

    Raises
    ------
        ValueError
            If not enought parameters are specified 
            or the input shape is inconsistent 


    """
    def __init__(self, input_shape=None, lb=None, ub=None):
        # Handle LB
        lb_ = None
        if lb is None and input_shape is None:
            raise ValueError('Not enough information for the lower bounds')
        try:
            # Array lb parameter
            shape = lb.shape
            if input_shape is not None and input_shape != shape:
                raise ValueError('Inconsistent LB shape and explicit shape')
            lb_ = lb
        except AttributeError:
            if input_shape is None:
                raise ValueError('Not enough information for the lower bounds')
            if lb is None:
                # None lb
                lb_ = np.full(input_shape, float('-inf'))
            else:
                # Scalar lb
                lb_ = np.full(input_shape, lb)
        # Handle UB
        ub_ = None
        if ub is None and input_shape is None:
            raise ValueError('Not enough information for the upper bounds')
        try:
            # Array ub parameter
            shape = ub.shape
            if input_shape is not None and input_shape != shape:
                raise ValueError('Inconsistent UB shape and explicit shape')
            ub_ = ub
        except AttributeError:
            if input_shape is None:
                raise ValueError('Not enough information for the upper bounds')
            if ub is None:
                # None ub
                ub_ = np.full(input_shape, float('+inf'))
            else:
                # Scalar ub
                ub_ = np.full(input_shape, ub)
        # Check bound consistency
        if np.any(lb_ > ub_):
            raise ValueError('Some LBs are higher than the LBs')
        # Store original bounds
        self._orig_lb = lb_.copy()
        self._orig_ub = ub_.copy()
        # Set the bounds
        super(DNRInput, self).__init__(lb_, ub_)

    def connected(self, *idx): #TODO
        """ Get the neurons connected to a specific one in the current layer

        Parameters
        ----------
            idx : (int, int)
                Coordinates of a specific neuron
        
        Returns
        -------
            None

        """
        assert(len(idx) == 1)
        return

    def ltype(self):
        """ Get type of the layer

        Returns
        -------
            Layer type : String
                String identifying the type of the layer

        """
        return 'input'

    def reset_bounds(self):
        """ Reset lower and upper bounds of the layer

        Returns
        -------
            None

        """
        self.lb_ = self._orig_lb.copy()
        self.ub_ = self._orig_ub.copy()

    def __repr__(self):
        """ Representation of the layer

        Returns
        -------
            Representation : string
                String representing the layer

        """
        return '[%s] ' % self.ltype() + \
                ' '.join('%s' % str(n) for n in self.neurons())

# ============================================================================
# Dense layer
# ============================================================================

class DNRDense(DNRActLayer):
    """ Class used to shape the dense layer of the neural network

    Attributes
    ----------
        lb_ : list(float)
            List of lower bounds for the input layer
        ub_ : list(float) 
            list of upper bounds for the input layer
        idx_ : int
            Index of the input layer
        net_ : string
            name of the neural network
        weights_ : list(float)
            List of weight associated 
            to the neurons in the layer
        bias_ : list(float) 
            List of bias associated 
            to the neurons in the layer
        activation_ : :obj:`keras.activations` 
            Activation function 

    Parameters
    ----------
        weights : list(float) 
            List of weight associated 
            to the neurons in the layer
        bias : list(float)
            List of bias associated 
            to the neurons in the layer
        activation : :obj:`keras.activations` 
            Activation function 

    Raises
    ------
        ValueError
            If bias and weight amatrixes are inconsistent

    """
    def __init__(self, weights, bias, activation):
        # Set the bounds
        if weights.shape[1] != len(bias):
            raise ValueError('Inconsistent weight matrix and bias vector sizes')
        self.weights_ = weights
        self.bias_ = bias
        self.activation_ = activation
        ylb_ = np.full(len(bias), -float('inf'))
        yub_ = np.full(len(bias), float('inf'))
        lb_ = act_eval(activation, ylb_)
        ub_ = act_eval(activation, yub_)
        super(DNRDense, self).__init__(lb_, ub_, ylb_, yub_)

    def connected(self, *idx): # TODO
        """ Get the neurons connected to a specific one in the current layer

        Parameters
        ----------
            idx : (int, int)
                Coordinates of a specific neuron

        Yields
        ------
            Index : (int, int)
                Coordinates to the connected neuron

        """
        assert(len(idx) == 1)
        it = np.nditer(self.weights_[...,idx[0]], flags=['multi_index'])
        pidx = (self.idx_[0]-1,)
        while not it.finished:
            yield pidx + it.multi_index[:-1]
            it.iternext()

    def weights(self, *idx):
        """ Get the weights associated to the neurons in a specific layer

        Parameters
        ----------
            idx : (int, int)
                Index of the layer

        Yields
        ------
            Weight : float
                Next weight in the layer

        """
        assert(len(idx) == 1)
        it = np.nditer(self.weights_[...,idx[0]])
        while not it.finished:
            yield it[0]
            it.iternext()

    def bias(self, *idx):
        """ Get the bias associated to a specific neuron in the layer

        Parameters
        ----------
            idx : (int, int)
                Index of the layer

        Returns
        -------
            Bias : float
                Value corresponding to the bias of the 
                current neuron

        """
        assert(len(idx) == 1)
        return self.bias_[idx[0]]

    def activation(self):
        """ Get activation function of the layer

        Returns
        -------
            Activation function : :obj:`keras.activations`
                Activation function of the layer

        """
        return self.activation_

    def evaluate(self, data):
        """ Evalueates the output of the layer given a input
        
        Parameteres
        -----------
            data : list(float) 
                Input for the layer

        Returns
        -------
            Layer output : float
                Output of the layer

        """
        yeval = np.dot(self.weights_.T, data) + self.bias_
        return act_eval(self.activation_, yeval), yeval

    def ltype(self):
        """ Get type of the layer

        Returns
        -------
            Layer type : string
                String identifying the type of the layer

        """
        return 'dense,%s' % self.activation_

    def reset_bounds(self):
        """ Reset lower and upper bounds of the layer

        Returns
        -------
            None

        """
        self.ylb_ = np.full(len(self.bias_), -float('inf'))
        self.yub_ = np.full(len(self.bias_), float('inf'))
        self.lb_ = act_eval(self.activation_, self.ylb_)
        self.ub_ = act_eval(self.activation_, self.yub_)

    def __repr__(self):
        """ Representation of the layer

        Returns
        -------
            Representation : string
                String representing the layer

        """
        return '[%s] ' % self.ltype() + \
            ' '.join('%s' % str(n) for n in self.neurons())


# ============================================================================
# Network
# ============================================================================

class DNRNet(object):
    """ Class used to shape the neural network
    
    Attributes
    ----------
        layers_ : list(:obj:`eml.net.describe.DNRLayers`)
            List of layers composing the network
    """
    def __init__(self):
        super(DNRNet, self).__init__()
        self.layers_ = []

    def add(self, layer):
        """ Add layer to the network 

        Parameters
        ----------
            layer : :obj:`eml.net.describe.DNRLayers`
                Layer to add 

        Returns
        -------
            None

        """
        layer.idx_ = (len(self.layers_),)
        layer.net_ = self
        self.layers_.append(layer)

    def nlayers(self):
        """ Get number of layers

        Returns
        -------
            Number of layers : int
                Integer representing the number of layers

        """
        return len(self.layers_)

    def layers(self):
        """ Get layers in the network

        Yields
        ------
            Layer : :obj:`eml.net.describe.DNRLayers`
                Next layer in the network
        
        """
        for i in range(len(self.layers_)):
            yield self.layers_[i]

    def layer(self, idx):
        """ Get layer with specific index

        Parameters
        ----------
            idx : int
                Layer index

        Returns
        -------
            Layer : :obj:`eml.net.describe.DNRLayer`
                Layer located at the given coordinates

        """
        return self.layers_[idx]

    def neuron(self, *idx):
        """ Get neuron in a specific location in the network

        Parameters
        -----------
            idx : (int, int)
                Index of the neuron, (layer, neuron)

        Returns
        -------
            Neuron : :obj:`eml.net.describe.DNRNeuron`
                Neuron located at the given coordinates

        """
        if len(idx) == 1:
            idx = idx[0]
        return self.layer(idx[0]).neuron(idx[1:])

    def neurons(self):
        """ Get all neurons in the network

        Yields
        ------
            Neuron : :obj:`eml.net.describe.DNRNeuron`
                Next neuron in the network
            
        """
        for layer in self.layers():
            for neuron in layer.neurons():
                yield neuron

    def size(self):
        """ Get size of the network

        Returns
        -------
            Number of neurons : int
                Integer describing the size of the network

        """
        return sum(l.size() for l in self.layers_)

    def reset_bounds(self):
        """ Reset lower and upper bound in the network

        Returns
        -------
            None

        """
        for layer in self.layers():
            layer.reset_bounds()

    def evaluate(self, data):
        """ Evalueates the output of the neural netwokr 

        Parameters
        ----------
            data : list(float) 
                Input for the network

        Returns
        -------
            Network output : float
                output of the network given the input

        """
        res, yres = [data], [None]
        for layer in self.layers_[1:]:
            if issubclass(type(layer), DNRActLayer):
                xeval, yeval = layer.evaluate(res[-1])
            else:
                xeval = layer.evaluate(res[-1])
                yeval = None
            res.append(xeval)
            yres.append(yeval)
        res = DNREvaluation(self, res, yres)
        return res

    def __repr__(self):
        """ Representation of the network

        Returns
        -------
            Representation : string
                String representing the network

        """
        return '\n'.join('%s' % str(l) for l in self.layers_)


# ============================================================================
# Network evaluation
# ============================================================================

class DNREvaluation(object):
    """ Class used to shape the neural network evaluator

    This class evaluates the results of a neural network
    buil according to the model at :obj:`eml.net.describe.DNRNet`

    Attributes
    ----------
        net_ : :obj:`eml.net.describe.DNRNet`
            Neural network
        evals_ : list(list(float))
            Evaluations of a layer
        yeavals_ : list(list(float))
            Evaluations of an activation layer

    Parameters
    ----------
        net :  :obj:`eml.net.describe.DNRNet`
            Neural netwokr
         evals : list(list(float))
            Evaluations of a layer
        yeavals : list(list(float))
            Evaluations of an activation layer

    
    """
    def __init__(self, net, evals, yevals):
        super(DNREvaluation, self).__init__()
        self.net_ = net
        self.evals_ = evals
        self.yevals_ = yevals

    def layer(self, idx): 
        """ Get specific evaluation of a layer

        Parameters
        ----------
            idx : int
                Index of the layer

        Returns
        -------
            Evaluations : list(float)
                Evaluations of the specified layer

        """
        return self.evals_[idx]

    def ylayer(self, idx):
        """ Get specific evaluation for activation layer

        Parameters
        ----------
            idx : int
                Index of the layer

        Returns
        -------
            Evaluations : list(float)
                Evaluations of the specified activation layer

        """
        return self.yevals_[idx]

    def xval(self, *idx):
        """ Get evaluation of a specific neuron 
        
        Parameters
        ----------
            idx : (int, int)
                Index of the neuron

        Returns
        ------- 
            Evaluation : float
                Evaluation of the neuron

        """
        assert(len(idx) > 1)
        return self.evals_[idx[0]][idx[1:]]

    def yval(self, *idx):
        """ Get evaluation of a specific activation neuron 

        Parameters
        ----------
            idx : (int, int)
                Index of the activation neuron

        Returns
        ------- 
            Evaluation : float
                Evaluation of the activation neuron
                
        """
        assert(len(idx) > 1)
        if self.yevals_[idx[0]] is None:
            raise ValueError('No Y evaluation for this layer')
        return self.yevals_[idx[0]][idx[1:]]

    def __repr__(self):
        """ Representation of the evaluator

        Returns
        -------
            Representation : string 
                String representing the evaluator

        """
        s = ''
        for k, layer in enumerate(self.net_.layers()):
            if k > 0:
                s += '\n'
            s += '[%s]' % layer.ltype()
            for neuron in layer.neurons():
                nidx = neuron.idx()
                # Add y evaluation
                s += ' %s:' % str(nidx)
                if self.yevals_[k] is not None:
                    s += '%.3f/' % self.yevals_[k][nidx[1:]]
                s += '%.3f' % self.evals_[k][nidx[1:]]
        return s


