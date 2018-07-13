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
    if activation == 'relu':
        return np.maximum(0, value)
    elif activation == 'linear':
        return value
    else:
        raise ValueError('Invalid activation function "%s"' % activation)


class DNRLayer(object):
    """docstring for DNRInput"""
    def __init__(self, lb, ub):
        super(DNRLayer, self).__init__()
        assert(lb.shape == ub.shape)
        self.lb_ = lb
        self.ub_ = ub
        self.idx_ = 0
        self.net_ = None

    def network(self):
        return self.net_

    def size(self):
        return len(self.lb_)
    
    def neuron(self, *idx):
        if len(idx) == 1:
            idx = idx[0]
        return DNRNeuron(self, idx)

    def neurons(self):
        it = np.nditer(self.lb_, flags=['multi_index'])
        while not it.finished:
            yield DNRNeuron(self, it.multi_index)
            it.iternext()

    def lb(self):
        return self.lb_[:]

    def update_lb(self, value, tol=1e-4):
        mask = (self.lb_  < value-tol)
        self.lb_[mask] = value
        return mask

    def ub(self):
        return self.ub_[:]

    def update_ub(self, value, tol=1e-4):
        mask = (self.ub_ > value+tol)
        self.ub_[mask] = value
        return mask

    def reset_bounds(self):
        raise NotImplementedError('This method should be implemented in subclasses')

    def connected(self, *idx):
        raise NotImplementedError('This method should be implemented in subclasses')

    def ltype(self):
        raise NotImplementedError('This method should be implemented in subclasses')

    def __repr__(self):
        return ' '.join('%s' % str(n) for n in self.neurons())


class DNRNeuron(object):
    """docstring for DNRInputNeuron"""
    def __init__(self, layer, idx):
        super(DNRNeuron, self).__init__()
        self.layer_ = layer
        self.idx_ = idx

    def layer(self):
        return self.layer_

    def network(self):
        return self.layer_.net_

    def network(self):
        return self.layer_.network()

    def idx(self):
        return self.layer_.idx_ + self.idx_

    def lb(self):
        return self.layer_.lb_[self.idx_]

    def update_lb(self, value, tol=1e-4):
        if self.layer_.lb_[self.idx_] < value-tol:
            self.layer_.lb_[self.idx_] = value
            return True
        return False

    def ub(self):
        return self.layer_.ub_[self.idx_]

    def update_ub(self, value, tol=1e-4):
        if self.layer_.ub_[self.idx_] > value+tol:
            self.layer_.ub_[self.idx_] = value
            return True
        return False

    def connected(self):
        return self.layer_.connected(self.idx_)

    def __repr__(self):
        return '%s:[%.3f, %.3f]' % (self.idx(), self.lb(), self.ub())


class DNRActLayer(DNRLayer):
    """docstring for DNRInput"""
    def __init__(self, lb, ub, ylb, yub):
        assert(lb.shape == ub.shape)
        assert(ylb.shape == yub.shape)
        super(DNRActLayer, self).__init__(lb, ub)
        self.ylb_ = ylb
        self.yub_ = yub

    def neuron(self, *idx):
        if len(idx) == 1:
            idx = idx[0]
        return DNRActNeuron(self, idx)

    def neurons(self):
        it = np.nditer(self.lb_, flags=['multi_index'])
        while not it.finished:
            yield DNRActNeuron(self, it.multi_index)
            it.iternext()

    def ylb(self):
        return self.ylb_[:]

    def update_ylb(self, value, tol=1e-4):
        mask = (self.ylb_  < value-tol)
        self.ylb_[mask] = value
        return mask

    def yub(self):
        return self.yub_[:]

    def update_yub(self, value, tol=1e-4):
        mask = (self.yub_ > value+tol)
        self.yub_[mask] = value
        return mask

    def reset_bounds(self):
        raise NotImplementedError('This method should be implemented in subclasses')

    def weights(self, *idx):
        raise NotImplementedError('This method should be implemented in subclasses')

    def bias(self, *idx):
        raise NotImplementedError('This method should be implemented in subclasses')

    def evaluate(self, data):
        raise NotImplementedError('This method should be implemented in subclasses')


class DNRActNeuron(DNRNeuron):
    """docstring for DNRInputNeuron"""
    def __init__(self, layer, idx):
        super(DNRActNeuron, self).__init__(layer, idx)

    def ylb(self):
        return self.layer_.ylb_[self.idx_]

    def update_ylb(self, value, tol=1e-4):
        if self.layer_.ylb_[self.idx_] < value-tol:
            self.layer_.ylb_[self.idx_] = value
            return True
        return False

    def yub(self):
        return self.layer_.yub_[self.idx_]

    def update_yub(self, value, tol=1e-4):
        if self.layer_.yub_[self.idx_] > value+tol:
            self.layer_.yub_[self.idx_] = value
            return True
        return False

    def weights(self):
        return self.layer_.weights(self.idx_)

    def bias(self):
        return self.layer_.bias(self.idx_)

    def activation(self):
        return self.layer_.activation_

    def __repr__(self):
        return '%s:[%.3f, %.3f]/[%.3f, %.3f]' % (self.idx(),
                self.ylb(), self.yub(), self.lb(), self.ub())

# ============================================================================
# Input layer
# ============================================================================

class DNRInput(DNRLayer):
    """docstring for DNRInput"""
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

    def connected(self, *idx):
        assert(len(idx) == 1)
        return

    def ltype(self):
        return 'input'

    def reset_bounds(self):
        self.lb_ = self._orig_lb.copy()
        self.ub_ = self._orig_ub.copy()

    def __repr__(self):
        return '[%s] ' % self.ltype() + \
                ' '.join('%s' % str(n) for n in self.neurons())

# ============================================================================
# Dense layer
# ============================================================================

class DNRDense(DNRActLayer):
    """docstring for DNRInput"""
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

    def connected(self, *idx):
        assert(len(idx) == 1)
        it = np.nditer(self.weights_[...,idx[0]], flags=['multi_index'])
        pidx = (self.idx_[0]-1,)
        while not it.finished:
            yield pidx + it.multi_index[:-1]
            it.iternext()

    def weights(self, *idx):
        assert(len(idx) == 1)
        it = np.nditer(self.weights_[...,idx[0]])
        while not it.finished:
            yield it[0]
            it.iternext()

    def bias(self, *idx):
        assert(len(idx) == 1)
        return self.bias_[idx[0]]

    def activation(self):
        return self.activation_

    def evaluate(self, data):
        yeval = np.dot(self.weights_.T, data) + self.bias_
        return act_eval(self.activation_, yeval), yeval

    def ltype(self):
        return 'dense,%s' % self.activation_

    def reset_bounds(self):
        self.ylb_ = np.full(len(self.bias_), -float('inf'))
        self.yub_ = np.full(len(self.bias_), float('inf'))
        self.lb_ = act_eval(self.activation_, self.ylb_)
        self.ub_ = act_eval(self.activation_, self.yub_)

    def __repr__(self):
        return '[%s] ' % self.ltype() + \
                ' '.join('%s' % str(n) for n in self.neurons())


# ============================================================================
# Network
# ============================================================================

class DNRNet(object):
    """docstring for DNRNet"""
    def __init__(self):
        super(DNRNet, self).__init__()
        self.layers_ = []

    def add(self, layer):
        layer.idx_ = (len(self.layers_),)
        layer.net_ = self
        self.layers_.append(layer)

    def nlayers(self):
        return len(self.layers_)

    def layers(self):
        for i in range(len(self.layers_)):
            yield self.layers_[i]

    def layer(self, idx):
        return self.layers_[idx]

    def neuron(self, *idx):
        if len(idx) == 1:
            idx = idx[0]
        return self.layer(idx[0]).neuron(idx[1:])

    def neurons(self):
        for layer in self.layers():
            for neuron in layer.neurons():
                yield neuron

    def size(self):
        return sum(l.size() for l in self.layers_)

    def reset_bounds(self):
        for layer in self.layers():
            layer.reset_bounds()

    def evaluate(self, data):
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
        return '\n'.join('%s' % str(l) for l in self.layers_)


# ============================================================================
# Network evaluation
# ============================================================================

class DNREvaluation(object):
    """docstring for DNREvaluation"""
    def __init__(self, net, evals, yevals):
        super(DNREvaluation, self).__init__()
        self.net_ = net
        self.evals_ = evals
        self.yevals_ = yevals

    def layer(self, idx):
        return self.evals_[idx]

    def ylayer(self, idx):
        return self.yevals_[idx]

    def xval(self, *idx):
        assert(len(idx) > 1)
        return self.evals_[idx[0]][idx[1:]]

    def xval(self, *idx):
        assert(len(idx) > 1)
        if self.yevals_[idx[0]] is None:
            raise ValueError('No Y evaluation for this layer')
        return self.yevals_[idx[0]][idx[1:]]

    def __repr__(self):
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


