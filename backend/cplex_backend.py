#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import sys
import docplex.mp.model as cpx

from eml.backend import base

class CplexBackend(base.Backend):
    """ Backend for CPLEX solver

    Attributes
    ---------
        _ml_tol  : float
            Tollerance 

    Parameters
    ----------
        ml_tol :float)
            Tollerance

    """
    def __init__(self, ml_tol=1e-4):
        self._ml_tol = ml_tol
        super(CplexBackend, self).__init__()

    def const_eps(self, mdl):
        """ Get tollerance

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 

        Returns
        -------
            Tollerance : float
                Tollerance 

        """ 
        return self._ml_tol

    def var_cont(self, mdl, lb, ub, name=None):
        """ Creates continuous variable in the model 

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            lb : float)
                Lower bound of the variable 
            ub :float
                Upper bound of the variable 
            name : string 
                Name of the variable (default None)

        Returns
        -------
            Continuos Variable : :obj:`docplex.mp.linear.Var``
                Continuos variable with specified bounds and name

        """
        # Convert bounds in a cplex friendly format
        lb = lb if lb != -float('inf') else -mdl.infinity()
        ub = ub if ub != float('inf') else mdl.infinity()
        # Build the variable
        return mdl.continuous_var(lb=lb, ub=ub, name=name)

    def var_bin(self, mdl, name=None):
        """ Creates continuous variable in the model

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            name : string)
                Name of the variable (default None)

        Returns
        ------- 
            Binary Variable : :obj:`docplex.mp.linear.Var``
                Binary Variable

        """
        return mdl.binary_var(name=name)

    def xpr_scalprod(self, mdl, coefs, terms):
        """ Scalar product of varibles and coefficients

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            coefs : list(float)
                List of coefficients 
            terms : list(:obj:docplex.mp.linear.Var]): 
                List of variables

        Returns
        -------
            Linear Expression : :obj:`docplex.mp.LinearExpr()`
                Linear expression representing the linear combination 
                of terms and coefficients or 0

        """
        return sum(c * x for c, x in zip(coefs, terms))

    def xpr_sum(self, mld, terms):
        """ Sum of variables 

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            terms : list(:obj:docplex.mp.linear.Var)
                List of variables

        Returns
        -------
            Linear Expression : :obj:`docplex.mp.LinearExpr()`
                Linear expression representing the sum of all
                the term in input

        """
        return sum(terms)

    def xpr_eq(self, mdl, left, right):
        """ Creates an equality constraint between two variables

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            left : :obj:docplex.mp.linear.Var
                Variable
            right : :obj:docplex.mp.linear.Var
                Variable

        Returns
        -------
            Equality constraint : :obj:`docplex.mp.constr.LinearConstraint`
                Equality contraint between the two variables in input 
        
        """
        return left == right

    def cst_eq(self, mdl, left, right, name=None):
        """ Add to the model equality constraint between two variables

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            left : :obj:docplex.mp.linear.Var
                Variable
            right : :obj:docplex.mp.linear.Var
                Variable
            name : string 
                Name of the constraint 

        Returns
        -------
            Equality constraint : :obj:`docplex.mp.constr.LinearConstraint`
                Equality contraint between the two variables in input 


        """
        return mdl.add_constraint(left == right, ctname=name)

    def cst_leq(self, mdl, left, right, name=None):
        """ Add to the model a lowe or equal constraint between two variables

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            left : :obj:docplex.mp.linear.Var
                Variable
            right : :obj:docplex.mp.linear.Var
                Variable
            name : string 
                Name of the constraint 

        Returns
        -------
            Lower or equal constraint : :obj:`docplex.mp.constr.LinearConstraint`
                Lowe or equal contraint between the two variables in input 


        """
        return mdl.add_constraint(left <= right, ctname=name)

    def cst_indicator(self, mdl, trigger, val, cst, name=None):
        """ Add an indicator to the model

        An indicator constraint links (one-way) the value of a
        binary variable to the satisfaction of a linear constraint

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            trigger : :obj:`docplex.mp.Var`
                Binary Variable 
            val : int
                Active value, used to trigger the satisfaction
                of the constraint
            cst : :obj:`docplex.mp.constr.LinearConstraint`
                Linear constraint 
            name : string 
                Name of the constraint 

        Returns
        -------
            Indicator constraint : :obj:`docplex.mp.constr.IndicatorConstraint`
                Indicator constraint between the trigger and the linear
                constraint in input

        """ 
        return mdl.add_indicator(trigger, cst, val, name=name)

    def get_obj(self, mdl):
        """ Returns objextive expression 

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 

        Returns
        -------
            Objective and expression : (string, )
                'min' if the objective function is to be minimized,
                'max otherwise. 
                The expression repesenting the objective function

        """
        sense = 'min' if mdl.is_minimized() else 'max'
        xpr = mdl.get_objective_expr()
        return sense, xpr

    def set_obj(self, mdl, sense, xpr):
        """ Sets the objective function 

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model 
            sense : string 
                Represents the objective, 'min' or 'max'
            xpr : 
                Expression representing the objective function

        Returns
        -------
            None

        """
        mdl.set_objective(sense, xpr)

    def solve(self, mdl, timelimit):
        """ Solves the problem

        Parameters
        -----------
            mdl : :obj:`docplex.mp.model.Model`
                            Cplex model 
            timelimit : int
                time limit in seconds for the solver

        Returns
        -------
            Solution : :obj:`docplex.mp.solution.SolveSolution`
                A solution if the problem is feasible, the status of the
                of the solver otherwise

        """
        mdl.set_time_limit(max(1, timelimit))
        res = mdl.solve()
        stime = mdl.solve_details.time
        status = 'infeasible' if res is None else res.solve_details.status
        obj = None if res is None else res.objective_value
        bound = mdl.solve_details.best_bound
        lres = {'status':status, 'obj': obj, 'bound': bound,
                'time': stime}
        return lres

    # def add_neuron(self, neuron, **args):
    #     # Cache some attribute
    #     mdl = self._mdl
    #     # Obtain cplex-friendly bounds
    #     lb, ub = neuron.lb(), neuron.ub()
    #     lb = lb if lb != -float('inf') else -mdl.infinity()
    #     ub = ub if ub != float('inf') else mdl.infinity()
    #     # --------------------------------------------------------------------
    #     # Build a variable for the model output
    #     # --------------------------------------------------------------------
    #     idx = neuron.idx()
    #     x = mdl.continuous_var(lb=lb, ub=ub, name='%s_x%s' % (self._name, str(idx)))
    #     self.x_add('x'.format(self._name), idx, x)
    #     # --------------------------------------------------------------------
    #     # Check whether this is a neuron with an activation function
    #     # --------------------------------------------------------------------
    #     net = neuron.network()
    #     if issubclass(neuron.__class__, describe.DNRActNeuron):
    #         # Build an expression for the neuron activation
    #         yterms = [neuron.bias()]
    #         for pidx, wgt in zip(neuron.connected(), neuron.weights()):
    #             prdx = self.x_get('x', pidx)
    #             yterms.append(prdx * wgt)
    #         y = mdl.sum(yterms)
    #         self.x_add('y', idx, y)
    #         # TODO add the redundant constraints by Sanner
    #         # TODO add bounding constraints on the y expression
    #         # ----------------------------------------------------------------
    #         # Introduce the csts and vars for the activation function
    #         # ----------------------------------------------------------------
    #         act = neuron.activation()
    #         if act == 'relu':
    #             ylb, yub = neuron.ylb(), neuron.yub()
    #             # Trivial case 1: the neuron is always active
    #             if ylb >= 0:
    #                 mdl.add_constraint(x == y)
    #             # Trivial case 1: the neuron is always inactive
    #             elif yub <= 0:
    #                 mdl.add_constraint(x == 0)
    #             # Handle the non-trivial case
    #             else:
    #                 # Enfore the natural bound on the neuron output
    #                 # NOTE if interval based reasoning has been used to
    #                 # compute bounds, this will be always redundant
    #                 x.lb = max(0, lb)
    #                 # Introduce a binary activation variable
    #                 z = mdl.binary_var(name='%s_z%s' % (self._name, str(idx)))
    #                 self.x_add('z', idx, z)
    #                 # Introduce a slack variable
    #                 s = mdl.continuous_var(ub=-ylb, name='%s_s%s' % (self._name, str(idx)))
    #                 self.x_add('s', idx, s)
    #                 # Buid main constraint
    #                 cst = mdl.add_constraint(x - s == y, ctname='%s_r0%s' % (self._name, str(idx)))
    #                 # Build indicator constraints
    #                 mdl.add_indicator(z, s == 0, 1, name='%s_r1%s' % (self._name, str(idx)))
    #                 mdl.add_indicator(z, x == 0, 0, name='%s_r2%s' % (self._name, str(idx)))
    #         elif act == 'linear':
    #             mdl.add_constraint(x == y, ctname='%s_l%s' % (self._name, str(idx)))
    #         else:
    #             raise ValueError('Unsupported "%s" activation function' % act)


    # def neuron_bounds(self, neuron, timelimit=None, verbose=0, **args):
    #     super(CplexBackend, self).neuron_bounds(neuron)
    #     # Prepare some data structures
    #     idx, mdl, net = neuron.idx(), self._mdl, neuron.network()
    #     ttime, bchg = 0, False
    #     if issubclass(neuron.__class__, describe.DNRActNeuron):
    #         act = neuron.activation()
    #     else:
    #         act = None
    #     # --------------------------------------------------------------------
    #     # Store objective state
    #     # --------------------------------------------------------------------
    #     original_obj = self._mdl.get_objective_expr()
    #     original_min = self._mdl.is_minimize()
    #     # --------------------------------------------------------------------
    #     # Compute an upper bound
    #     # --------------------------------------------------------------------
    #     # Choose the correct objective function
    #     mdl.set_objective('max', self.x_get('x', idx))
    #     # Solve the problem and extract the best bound
    #     res = mdl.solve()
    #     # Extract the bound
    #     if res.solve_details.status == 'optimal':
    #         ub = res.objective_value
    #     else:
    #         ub = mdl.solve_details.best_bound
    #     ttime += mdl.solve_details.time
    #     # Enforce the bound
    #     # NOTE for activation neurons, this is an _activation_ bound!
    #     if act is not None:
    #         neuron.update_yub(ub)
    #         neuron.update_ub(describe.act_eval(act, ub))
    #     else:
    #         neuron.update_ub(ub)
    #     # --------------------------------------------------------------------
    #     # Compute a lower bound
    #     # --------------------------------------------------------------------
    #     # Choose the correct objective function
    #     if act == 'relu' and self.x_has('s', idx):
    #         mdl.set_objective('min', -self.x_get('s', idx))
    #     elif act == 'linear':
    #         mdl.set_objective('min', self.x_get('x', idx))
    #     else:
    #         mdl.set_objective('min', self.x_get('x', idx))
    #     # Solve the problem and extract the best bound
    #     res = mdl.solve()
    #     # Extract the bound
    #     if res.solve_details.status == 'optimal':
    #         lb = res.objective_value
    #     else:
    #         lb = mdl.solve_details.best_bound
    #     ttime += mdl.solve_details.time
    #     # Enforce the bound
    #     # NOTE for activation neurons, this is an _activation_ bound!
    #     if issubclass(neuron.__class__, describe.DNRActNeuron):
    #         neuron.update_ylb(lb)
    #         neuron.update_lb(describe.act_eval(act, lb))
    #     else:
    #         neuron.update_lb(lb)
    #     # --------------------------------------------------------------------
    #     # Restore objective state
    #     # --------------------------------------------------------------------
    #     obj_dir = 'min' if original_min else 'max'
    #     mdl.set_objective(obj_dir, original_obj)
    #     # --------------------------------------------------------------------
    #     # Return results
    #     # --------------------------------------------------------------------
    #     return ttime, bchg

    def new_model(self, mdl=None, name=None):
        """ Creates a new model 

        Parameters
        ----------
            mdl : :obj:`docplex.mp.model.Model`
                Cplex model (default None)
            name : string
                Name of the model (default None)

        Returns
        -------
            Model : :obj:`docplex.mp.model.Model`
                Cplex model

        """
        return cpx.Model()

    
    # def update_lb():

    # def update_ub():
        

    # def set_model(self, mdl=None):
    #     # Clear the current model (if owned)
    #     if self._mdl_owned:
    #         self._mdl.end()
    #     # Build an internal model if necessary
    #     if mdl is None:
    #         self._mdl = cpx.Model()
    #         self._mdl_owned = True
    #     else:
    #         self._mdl = mdl
    #         self._mdl_owned = False

    # def __del__(self):
    #     if self._mdl_owned:
    #         self._mdl.end()

    # def __enter__(self):
    #     return self

    # def __exit(self):
    #     if self._mdl_owned:
    #         mdl.end()


def model_to_string(mdl):
    """ Returns a string representing the model
    
    Parameters
    ---------- 
        mdl : :obj:`docplex.mp.model.Model`
            Cplex model

    Returns
    -------
        Representation : string
            String representig the cplex model 

    """
    s = ''
    # Print objective
    if mdl.is_minimized():
        s += 'minimized: %s\n' % mdl.get_objective_expr()
    else:
        s += 'maximized: %s\n' % mdl.get_objective_expr()
    # Print all constraints
    s += 'subject to:\n'
    for cst in mdl.iter_constraints():
        s += '\t%s\n' % str(cst)
    # Print all variables
    s += 'with vars:\n'
    for var in mdl.iter_variables():
        s += '\t%f <= %s <= %f\n' % (var.lb, str(var), var.ub)
    return s
