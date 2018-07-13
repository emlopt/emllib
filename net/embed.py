#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np

# # Add the necessary paths
# path = os.path.abspath(os.path.dirname(__file__))
# if not path in sys.path:
#     sys.path.insert(1, path)
# del path
# import describe
# importlib.reload(describe)

from eml.net import describe
from eml import util

import importlib
importlib.reload(describe)
importlib.reload(util)

# class NetModelDesc:
#     def __init__(self, net, mdl, name):
#         self._net = net
#         self._mdl = mdl
#         self._name = name
#         self._exps = {}
#         self._neurons = set()

#     def store(self, xtype, xidx, val):
#         self._exps[(xtype,) + xidx] = val

#     def get(self, xtype, xidx):
#         return self._exps[(xtype,) + xidx]

#     def has(self, xtype, xidx):
#         return ((xtype,) + xidx) in self._exps

#     def expressions(self):
#         return self._exps

#     def name(self):
#         return self._name

#     def network(self):
#         return self._net

#     def model(self):
#         return self._mdl


def encode(bkd, net, mdl, net_in, net_out, name, verbose=0):
    # Scalar to vector output
    try:
        len(net_out)
    except:
        net_out = [net_out]
    # Build a model descriptor
    desc = util.ModelDesc(net, mdl, name)
    # Process the network layer by layer
    for k, layer in enumerate(net.layers()):
        # Add the layer to the solver wrapper
        for i, neuron in enumerate(layer.neurons()):
            # Add the neuron to the describe
            if verbose >= 1:
                print('Adding neuron %s' % str(neuron.idx()))
            if k == 0:
                x = net_in[i]
            elif k == net.nlayers()-1:
                x = net_out[i]
            else:
                x = None
            _add_neuron(bkd, desc, neuron, x=x)
    # Return the network descriptor
    return desc


def _add_neuron(bkd, desc, neuron, x=None):
    # Preliminary checks
    if neuron.network() != desc.ml_model():
        raise ValueError('The neuron does not belong to the correct network')
    # if neuron.idx() not in desc._neurons:
    #     desc._neurons.add(neuron.idx())
    # else:
    #     raise ValueError('The neuron has been already added')
    # Obtain current bounds
    lb, ub = neuron.lb(), neuron.ub()
    # Obtain network name and inner model
    sn, mdl = desc.name(), desc.model()
    # --------------------------------------------------------------------
    # Build a variable for the model output
    # --------------------------------------------------------------------
    idx = neuron.idx()
    if x is None:
        x = bkd.var_cont(mdl, lb, ub, '%s_x%s' % (sn, str(idx)))
    desc.store('x', idx, x)
    # --------------------------------------------------------------------
    # Check whether this is a neuron with an activation function
    # --------------------------------------------------------------------
    net = neuron.network()
    sn = desc.name()
    if issubclass(neuron.__class__, describe.DNRActNeuron):
        # Build an expression for the neuron activation
        coefs, yterms = [1], [neuron.bias()]
        for pidx, wgt in zip(neuron.connected(), neuron.weights()):
            prdx = desc.get('x', pidx)
            coefs.append(wgt)
            yterms.append(prdx)
        y = bkd.xpr_scalprod(mdl, coefs, yterms)
        desc.store('y', idx, y)
        # TODO add the redundant constraints by Sanner
        # TODO add bounding constraints on the y expression
        # ----------------------------------------------------------------
        # Introduce the csts and vars for the activation function
        # ----------------------------------------------------------------
        act = neuron.activation()
        if act == 'relu':
            ylb, yub = neuron.ylb(), neuron.yub()
            # Trivial case 1: the neuron is always active
            if ylb >= 0:
                bkd.cst_eq(mdl, x, y, '%s_l%s' % (sn, str(idx)))
            # Trivial case 1: the neuron is always inactive
            elif yub <= 0:
                bkd.cst_eq(mdl, x, 0, '%s_z%s' % (sn, str(idx)))
            # Handle the non-trivial case
            else:
                # Enfore the natural bound on the neuron output
                # NOTE if interval based reasoning has been used to
                # compute bounds, this will be always redundant
                x.lb = max(0, lb)
                # Introduce a binary activation variable
                z = bkd.var_bin(mdl, '%s_z%s' % (sn, str(idx)))
                desc.store('z', idx, z)
                # Introduce a slack variable
                s = bkd.var_cont(mdl, 0, -ylb, '%s_s%s' % (sn, str(idx)))
                desc.store('s', idx, s)
                # Buid main constraint
                left = bkd.xpr_scalprod(mdl, [1, -1], [x, s])
                bkd.cst_eq(mdl, left, y, '%s_r0%s' % (sn, str(idx)))
                # Build indicator constraints
                right = bkd.xpr_eq(mdl, s, 0)
                bkd.cst_indicator(mdl, z, 1, right, '%s_r1%s' % (sn, str(idx)))
                right = bkd.xpr_eq(mdl, x, 0)
                bkd.cst_indicator(mdl, z, 0, right, '%s_r2%s' % (sn, str(idx)))
        elif act == 'linear':
            bkd.cst_eq(mdl, x, y, '%s_l%s' % (sn, str(idx)))
        else:
            raise ValueError('Unsupported "%s" activation function' % act)
