#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

# ===========================================================================
# Encode Piecewise Linear Function
# ===========================================================================

# def encode_pwl(bkd, mdl, xvars, nodes, mode='sos2', name=None):
#     # Cache some fields
#     nnodes = len(nodes[0])
#     # Build node amount variables
#     tvars = []
#     for i in range(nnodes):
#         vname = None if name is None else '%s_a%d' % (name, i)
#         tvars.append(bkd.var_cont(mdl, lb=0, ub=1, name=vname))
#     # Ensure that we have a convex combination
#     bkd.cst_eq(mdl, bkd.xpr_sum(mdl, tvars), 1)
#     # Link the node amounts with the node values and the model variables
#     for xvar, lnodes in zip(xvars, nodes):
#         xpr = bkd.xpr_scalprod(mdl, lnodes, tvars)
#         bkd.cst_eq(mdl, xvar, xpr)
#     # Associate an integer "flag" variable to each node
#     ivars = []
#     for i in range(nnodes):
#         vname = None if name is None else '%s_f%d' % (name, i)
#         ivars.append(bkd.var_bin(mdl, name=vname))
#     # Connect the node amount and flag variables
#     for i in range(nnodes):
#         bkd.cst_leq(mdl, tvars[i], ivars[i])
#     # No more than 2 active flag variables
#     xpr = bkd.xpr_sum(mdl, ivars)
#     bkd.cst_leq(mdl, xpr, 2)
#     # Non contiguous flag variables cannot be simultanously active
#     for i in range(nnodes):
#         for j in range(i+2, nnodes):
#             xpr = bkd.xpr_sum(mdl, [ivars[i], ivars[j]])
#             bkd.cst_leq(mdl, xpr, 1)


# ===========================================================================
# Genericl model descriptor
# ===========================================================================

class ModelDesc:
    """ Class used to shape a model descriptor

    A model descriptor summarize the main feature a
    system, in this case will be used for empirical 
    model learning models

    Attributes
    ----------
        _ml :
            Machine learning model 
        _mdl : 
            Optimization model 
        _name : 
            Name of the model 
        _exps : string
            Expressions in the model

    Parameters
    ---------
        ml :
            Machine learning model 
        mdl : 
            Optimization model 
        name : string
            Name of the model 
    
    """
    def __init__(self, ml, mdl, name):
        self._ml = ml # ml model
        self._mdl = mdl # opt model 
        self._name = name
        self._exps = {}

    def store(self, xtype, xidx, val):
        """ Store expression 

        Parameters
        ----------
            xtype : string  
                Type of expression
            xidx : int
                Index we want to use to store
                the expression 
            val: 
                Value to store  

        """
        try:
            len(xidx)
        except:
            xidx = (xidx,) 
        self._exps[(xtype,) + xidx] = val

    def get(self, xtype, xidx):
        """ Get expression

        Parameters
        ----------
            xtype : string 
                Type of expression
            xidx : int
                Index we want to use to store
                the expression 
        
        Returns
        -------
            Expr : `generic type``
                Expression located by the coordinates

        """
        try:
            len(xidx)
        except:
            xidx = (xidx,) 
        return self._exps[(xtype,) + xidx]

    def has(self, xtype, xidx):
        """ Check if the model contains an expression given some coordinares
        
        Parameters
        ----------
            xtype : string 
                Type of expression
            xidx : int
                Index we want to use to store
                the expression 
        
        Returns
        -------
            Acknowledge : bool
                True if the expression is present, 
                False otherwise

        """ 
        try:
            len(xidx)
        except:
            xidx = (xidx,) 
        return ((xtype,) + xidx) in self._exps

    def expressions(self):
        """ Get all the expressions stored 

        Returns
        -------
            Expressions : dict(expr)
                Set of the expressions in the model 
        """
        return self._exps

    def name(self):
        """ Get name of the model 

        Returns
        -------
            Name : string
                Name of the model
        
        """
        return self._name

    def model(self):
        """ Get the optimization model

        Returns
        -------
            Optimization Model : :obj:`docplex.mp.model.Model`
                Combinatorial system 

        """
        return self._mdl

    def ml_model(self):
        """ Get machine learning model 

        Returns
        -------
            Machine learning model : `generic type`
                Machine learning model 
            
        """
        return self._ml

    def __repr__(self):
        """ Representation of the layer

        Returns
        -------
            Representation : string
                String representing the layer

        """
        s = '' 
        s += 'Model Name: ' + self._name + '\n'
        s += 'Machine Learning Model: ' + str(type(self._ml)) + '\n'
        s += 'Optimization Model: ' + str(type(self._mdl)) + '\n'
        # s += json.dumps(self._exps, indent=4, sort_keys=True)
        s += str(self._exps)
        return s