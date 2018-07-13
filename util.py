#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ===========================================================================
# Encode Piecewise Linear Function
# ===========================================================================

def encode_pwl(bkd, mdl, xvars, nodes, mode='sos2', name=None):
    # Cache some fields
    nnodes = len(nodes[0])
    # Build node amount variables
    tvars = []
    for i in range(nnodes):
        vname = None if name is None else '%s_a%d' % (name, i)
        tvars.append(bkd.var_cont(mdl, lb=0, ub=1, name=vname))
    # Ensure that we have a convex combination
    bkd.cst_eq(mdl, bkd.xpr_sum(mdl, tvars), 1)
    # Link the node amounts with the node values and the model variables
    for xvar, lnodes in zip(xvars, nodes):
        xpr = bkd.xpr_scalprod(mdl, lnodes, tvars)
        bkd.cst_eq(mdl, xvar, xpr)
    # Associate an integer "flag" variable to each node
    ivars = []
    for i in range(nnodes):
        vname = None if name is None else '%s_f%d' % (name, i)
        ivars.append(bkd.var_bin(mdl, name=vname))
    # Connect the node amount and flag variables
    for i in range(nnodes):
        bkd.cst_leq(mdl, tvars[i], ivars[i])
    # No more than 2 active flag variables
    xpr = bkd.xpr_sum(mdl, ivars)
    bkd.cst_leq(mdl, xpr, 2)
    # Non contiguous flag variables cannot be simultanously active
    for i in range(nnodes):
        for j in range(i+2, nnodes):
            xpr = bkd.xpr_sum(mdl, [ivars[i], ivars[j]])
            bkd.cst_leq(mdl, xpr, 1)


# ===========================================================================
# Genericl model descriptor
# ===========================================================================

class ModelDesc:
    def __init__(self, ml, mdl, name):
        self._ml = ml
        self._mdl = mdl
        self._name = name
        self._exps = {}

    def store(self, xtype, xidx, val):
        try:
            len(xidx)
        except:
            xidx = (xidx,)
        self._exps[(xtype,) + xidx] = val

    def get(self, xtype, xidx):
        try:
            len(xidx)
        except:
            xidx = (xidx,)
        return self._exps[(xtype,) + xidx]

    def has(self, xtype, xidx):
        try:
            len(xidx)
        except:
            xidx = (xidx,)
        return ((xtype,) + xidx) in self._exps

    def expressions(self):
        return self._exps

    def name(self):
        return self._name

    def tree(self):
        return self._tree

    def model(self):
        return self._mdl

    def ml_model(self):
        return self._ml
