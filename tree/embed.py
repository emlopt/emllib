#!/usr/bin/env python
# -*- coding: utf-8 -*-

from eml import util
from eml.tree.describe import DTNode

# ===========================================================================
# Utility functions
# ===========================================================================

def _extract_rules(root):
    """ Transforms the decistion tree into a set of rules

    Every rule represents a path from the root to a leaf. 
    The rule are composed by the attribute name, attribute type and threshold 
    test needed to go on along the path. Eventually, the last element of the 
    rule represents the class label of the leaf.

    Parameters
    ---------
        root : :obj:`eml.tree.describe.DTNode`
            Decision tree, build using sklearn

    Returns
    -------
        Rules : list(int, int, list(float, float), int)
            List containing all the possible paths leading from the root to a leaf

    """
    ris = []
    for child in root.get_children():
        ris += _aux(child)
    return ris


def _aux(node):
    base = [(node.attr_name(), node.attr_type(), node.attr_range())]
    if node.get_class() is not None:
        return [base + [node.get_class()]]
    else:
        ris = []
        for child in node.get_children():
            child_rules = _aux(child)
            for rule in child_rules:
                ris.append(base + rule)
    return ris
        

# ===========================================================================
# Backward encoding
# ===========================================================================

def encode_backward_implications(bkd, tree, mdl, tree_in, tree_out, name):
    """ Encode the decision tree in the backend

    Given a input and a output the tree is embeded into the optimization 
    problem.
    
    Parameters
    ----------
        bkd : :obj:`eml.backend.cplex_backend.CplexBackend`
            Cplex backend
        tree : :obj:`eml.tree.describe.DTNode``
            Decision tree
        mdl : :obj:`docplex.mp.model.Model`
            Cplex model 
        tree_in : list(:obj:`docplex.mp.linear.Var`)
            Input continuous variable 
        tree_out : :obj:`docplex.mp.linear.Var` 
            Output continuous variable 
        name : string
            Name fo the tree

    Returns
    -------
        Model Desciptor : :obj:`eml.util.ModelDesc`
            Descriptor of the instance of EML

    Raises
    ------
        ValueError
            If the threshold is in the 'right' branch or the tree
            has an output vector

    """
    # Build a model descriptor
    desc = util.ModelDesc(tree, mdl, name)
    # obtain the decision tree in rule format
    rules = _extract_rules(tree)
    nrules = len(rules)
    # ------------------------------------------------------------------------
    # Introduce a binary variable for each rule
    Z = []
    for k in range(nrules):
        if desc.has('path', k):
            zvar = desc.get('path', k)
        else:
            zvar = bkd.var_bin(mdl, '%s_p[%d]' % (name, k))
            desc.store('path', k, zvar)
        Z.append(zvar)
    # Only one rule can be active at a time
    bkd.cst_eq(mdl, bkd.xpr_sum(mdl, Z), 1)
    # ------------------------------------------------------------------------
    # Class assignment
    coefs = [r[-1] for r in rules]
    bkd.cst_eq(mdl, tree_out, bkd.xpr_scalprod(mdl, coefs, Z))
    # ------------------------------------------------------------------------
    # Collapse conditions on the same attribute for each rule
    crules = []
    for k, r in enumerate(rules):
        res = {}
        for aname, atype, (th1, th2) in r[:-1]:
            if aname not in res:
                res[aname] = (th1, th2)
            else:
                oth1, oth2 = res[aname]
                res[aname] = (max(oth1, th1), min(oth2, th2))
        crules.append(res)
    # ------------------------------------------------------------------------
    # Process all conditions in all rules
    built = set()
    for k, r in enumerate(rules):
        for aname, atype, (th1, th2) in r[:-1]:
            # If the constraint has already been built, then do nothing
            if (aname, th1, th2) in built:
                continue
            # Identify all rules that are based on this condition
            # Should work with implied rules, too
            # impl = [k for k, cr in enumerate(crules) if
            #         aname in cr and
            #         cr[aname][0] <= th1 and th2 < cr[aname][1]]
            based = [k for k, cr in enumerate(crules) if
                     aname in cr and
                     th1 <= cr[aname][0] and cr[aname][1] < th2]
            if th2 != float('inf'):
                M = tree.ub(aname)
                th = th2
                coefs = [1] + [M - th] * len(based)
                terms = [tree_in[aname]] + [Z[k] for k in based]
                cst = bkd.cst_leq(mdl, bkd.xpr_scalprod(mdl, coefs, terms), M)
            if th1 != -float('inf'):
                m = tree.lb(aname)
                th = th1 + bkd.const_eps(mdl)
                coefs = [1] + [m - th] * len(based)
                terms = [tree_in[aname]] + [Z[k] for k in based]
                cst = bkd.cst_geq(mdl, bkd.xpr_scalprod(mdl, coefs, terms), m)
    # Return the descriptor
    return desc
