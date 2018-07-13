#!/usr/bin/env python
# -*- coding: utf-8 -*-

from eml import util

# ===========================================================================
# Utility functions
# ===========================================================================

def _extract_rules(tree):
    # build a descriptor for the condition associated to this branch
    if tree.branch_aname is not None:
        base = [(tree.branch_aname, tree.branch_atype, tree.branch_label)]
    else:
        base = [] # no base if there is no branch
    # if the node is a class, end the rule with a "head"
    if tree.class_label is not None:
        return [base + [tree.class_label]]
    # otherwise, extend all the rules returned by the children
    else:
        my_rules = []
        for child in tree.children:
            child_rules = _extract_rules(child)
            for rule in child_rules:
                my_rules.append(base + rule)
        return my_rules


# ===========================================================================
# Backward encoding
# ===========================================================================

def encode_backward_implications(bkd, tree, mdl, tree_in, tree_out, name,
        verbose=0):
    # Build a model descriptor
    desc = util.ModelDesc(tree, mdl, name)
    sn = name # shortcut to the model name
    # obtain the decision tree in rule format
    rules = _extract_rules(tree)
    nrules = len(rules)
    # Quick argument check
    if not tree.thr_left:
        raise ValueError('Trees where the threshold goes in the right branch are not yet supported')
    try:
        if len(tree_out) > 1:
            raise ValueError('Trees with vector output are not yet supported')
        tree_out = tree_out[0]
    except:
        pass
    # ------------------------------------------------------------------------
    # Introduce a binary variable for each rule
    Z = []
    for k in range(nrules):
        if desc.has('path', k):
            zvar = desc.get('path', k)
        else:
            zvar = bkd.var_bin(mdl, '%s_p[%d]' % (sn, k))
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
    # for r in rules:
    #     print(r)
    built = set()
    for k, r in enumerate(rules):
        for aname, atype, (th1, th2) in r[:-1]:
            # If the constraint has already been built, then do nothing
            if (aname, th1, th2) in built:
                continue
            # Identify all rules that are based on this condition
            # TODO this should work with implied rules, too
            # impl = [k for k, cr in enumerate(crules) if
            #         aname in cr and
            #         cr[aname][0] <= th1 and th2 <= cr[aname][1]]
            based = [k for k, cr in enumerate(crules) if
                    aname in cr and
                    th1 <= cr[aname][0]and cr[aname][1] <= th2]
            # based = [k for k, cr in enumerate(rules) if
            #         aname in cr and
            #         cr[aname][0] == th1 and th2 == cr[aname][1]]
            # print aname, th1, th2
            # print [crules[k][aname] for k in based]
            # Post a constraint
            # print('-' * 30)
            # print(aname, atype, (th1, th2))
            # print(based)
            if th2 != float('inf'):
                M = tree.ub(aname)
                th = th2
                coefs = [1] + [M - th] * len(based)
                terms = [tree_in[aname]] + [Z[k] for k in based]
                cst = bkd.cst_leq(mdl, bkd.xpr_scalprod(mdl, coefs, terms), M)
                # print(cst)
            if th1 != -float('inf'):
                m = tree.lb(aname)
                th = th1 + bkd.const_eps(mdl)
                coefs = [1] + [m - th] * len(based)
                terms = [tree_in[aname]] + [Z[k] for k in based]
                cst = bkd.cst_geq(mdl, bkd.xpr_scalprod(mdl, coefs, terms), m)
                # print(cst)
            # raise RuntimeError('BABOON!')
    # Return the descriptor
    return desc
