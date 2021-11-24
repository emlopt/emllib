#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Add the necessary paths
# path = os.path.abspath(os.path.dirname(__file__))
# if not path in sys.path:
#     sys.path.insert(1, path)
# del path
# import describe
# importlib.reload(describe)

from eml import util
from eml.net import describe, embed


# ============================================================================
# Fast bounds via Interval Based Reasoning
# ============================================================================

def ibr_bounds(net):
    """ Internal Based Reasoning Bounding

    The bounds of the units in the neural network are updated based on the 
    values evaluated using the activation function
    
    Parameteres
    -----------
        net : :obj:`eml.net.describe.DNRNet`
            Neural network of interest 

    Returns 
    -------
        None

    """
    # Sequentially process layers
    for lidx, layer in enumerate(net.layers()):
        try:
            # Compute neuron activation
            for nrn in layer.neurons():
                # Compute the neuron activity
                yub, ylb = nrn.bias(), nrn.bias()
                for idx, wgt in zip(nrn.connected(), nrn.weights()):
                    prd = net.neuron(idx)
                    if wgt >= 0: 
                        yub += wgt * prd.ub()
                        ylb += wgt * prd.lb()
                    else:
                        yub += wgt * prd.lb()
                        ylb += wgt * prd.ub()
                # Apply the activation function
                lb = describe.act_eval(nrn.activation(), ylb)
                ub = describe.act_eval(nrn.activation(), yub)
                # Update the bounds
                nrn.update_ylb(ylb)
                nrn.update_yub(yub)
                nrn.update_lb(lb)
                nrn.update_ub(ub)
        except AttributeError as e:
            pass

# ============================================================================
# Forward bound tightening via Mixed Integer Linear Programming
# ============================================================================

def fwd_bound_tighthening(bkd, net=None, desc=None,
        timelimit=None, skip_layers=None, verbose=0):
    """ Forward bound tightening via Mixed Integer Linear Programming 
    

    Parameters
    ----------
        bkd : :obj:`eml.backend.cplex_backend.CplexBackend`
            Cplex backend
        net : obj:`eml.net.describe.DNRNet`
            Neural network of interest (default None)
        desc : :obj:`eml.util.ModelDesc`
            Model descriptor (default None)
        timelimit : int
            Time limit for the process (default None)
        skip_layer : int
            Skips bound tightening for the specified layer (default None)
        verbose : int
            if higher than 0 prints more info on the process (default 0)

    Returns
    -------
        Total time : int 
            Time used to perform bound tightening by the optimizer

    Raises
    ------
        ValueError
            Neither a model descriptor or a network where given in input

    """
    # Check args
    if (net is None and desc is None) or (net is not None and desc is not None):
        raise ValueError('Either a network or a network model descriptor should be passed ')
    # If no model descriptor is passed, one is built internally
    if net is not None:
        mdl = bkd.new_model()
        desc = util.ModelDesc(net, mdl, name='_tmp')
        build_neurons = True
    else:
        net = desc.ml_model()
        build_neurons = False
    # Process the network layer by layer
    ttime, nleft = 0, net.size()
    for k, layer in enumerate(net.layers()):
        # Add the layer to the solver wrapper
        for neuron in layer.neurons():
            # Add the neuron to the describe
            if build_neurons:
                if verbose >= 1:
                    print('Adding neuron %s' % str(neuron.idx()))
                embed._add_neuron(bkd, desc, neuron)
            # Do not compute the bounds for skipped layers
            if skip_layers is not None and layer.idx() in skip_layers:
                continue
            # Obtain a time limit for computing bounds
            if timelimit is not None:
                tlim = (timelimit - ttime) / nleft
            else:
                tlim = None
            # Compute bounds
            if verbose >= 1:
                print('Computing bounds for %s' % str(neuron.idx()))
            ltime, bchg = _neuron_bounds(bkd, desc, neuron, timelimit=tlim,
                    verbose=verbose)
            ttime += ltime
            nleft -= 1
    # Return total time
    return ttime


def _neuron_bounds(bkd, desc, neuron, timelimit):
    """ Bound tightening for neurons 
    
    Parameters
    ----------
        bkd : :obj:`eml.backend.cplex_backend.CplexBackend`
            Cplex backend
        desc : :obj:`eml.util.ModelDesc`
            Model Descriptor
        neuron : obj:`eml.net.describe.DNRNeuron`
            Neuron of interest
        timelimit : int
            Time limit to perform the process

    Returns
    -------
        Time Spend : int
            Time spend performing the bound tightening on the neuron
            by the optimizer

    Raises
    ------
        ValueError
            Neuron not in the current network 

    """
    # Preliminary checks
    if neuron.network() != desc.ml_model():
        raise ValueError('The neuron does not belong to the correct network')
    # Prepare some data structures
    idx, net = neuron.idx(), neuron.network()
    ttime = 0
    if issubclass(neuron.__class__, describe.DNRActNeuron):
        act = neuron.activation()
    else:
        act = None
    # Internal model
    mdl = desc.model()
    # --------------------------------------------------------------------
    # Store objective state
    # --------------------------------------------------------------------
    old_sense, old_obj = bkd.get_obj(mdl)
    # --------------------------------------------------------------------
    # Compute an upper bound
    # --------------------------------------------------------------------
    # Choose the correct objective function
    bkd.set_obj(mdl, 'max', desc.get('x', idx))
    # Solve the problem and extract the best bound
    res = bkd.solve(mdl, 0.5 * timelimit)
    # Extract the bound
    if res['status'] == 'optimal':
        ub = res['obj']
    else:
        ub = res['bound']
    ttime += res['time']
    # Enforce the bound
    # NOTE for activation neurons, this is an _activation_ bound!
    if act is not None:
        neuron.update_yub(ub)
        neuron.update_ub(describe.act_eval(act, ub))
    else:
        neuron.update_ub(ub)
    # --------------------------------------------------------------------
    # Compute a lower bound
    # --------------------------------------------------------------------
    # Choose the correct objective function
    if act == 'relu' and desc.has('s', idx):
        bkd.set_obj(mdl, 'min', -desc.get('s', idx))
    elif act == 'linear':
        bkd.set_obj(mdl, 'min', desc.get('x', idx))
    else:
        bkd.set_obj(mdl, 'min', desc.get('x', idx))
    # Solve the problem and extract the best bound
    res = bkd.solve(mdl, timelimit - ttime)
    # Extract the bound
    if res['status'] == 'optimal':
        lb = res['obj']
    else:
        lb = res['bound']
    ttime += res['time']
    # Enforce the bound
    # NOTE for activation neurons, this is an _activation_ bound!
    if issubclass(neuron.__class__, describe.DNRActNeuron):
        neuron.update_ylb(lb)
        neuron.update_lb(describe.act_eval(act, lb))
    else:
        neuron.update_lb(lb)
    # --------------------------------------------------------------------
    # Restore objective state
    # --------------------------------------------------------------------
    bkd.set_obj(mdl, old_sense, old_obj)
    # --------------------------------------------------------------------
    # Return results
    # --------------------------------------------------------------------
    return ttime
