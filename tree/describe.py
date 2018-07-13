#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# ===========================================================================
# Classes to model a decision tree
# ===========================================================================

class DTNode(object):
    '''
    Node in a decision tree
    '''

    # Attribute type codes
    attr_num, attr_sym = range(2)

    def __init__(self, root=None):
        # super(DTNode, self).__init__()
        if root is None:
            self._root = self
            self._atypes = {}
            self._albs = {}
            self._aubs = {}
        else:
            self._root = root
        # init node attributes
        self.aname = None
        self.atype = None
        # parent node
        self.parent = None
        # brach attributes
        self.branch_label = None
        self.branch_aname = None
        self.branch_atype = None
        # list of children
        self.children = []
        # define the class label (for leaves only)
        self.class_label = None
        # Should the threshold go in the left or right branch?
        self.thr_left = True

    def add_child(self, node, aname, atype, label):
        # TODO convert the assertions to exceptions
        # setup the parent-child relationship
        assert node.parent is None
        self.children.append(node)
        node.parent = self
        # setup the branch labels
        assert not node.atype == DTNode.attr_num or len(label) == 2
        assert not node.atype == DTNode.attr_num or label[0] < label[1]
        node.branch_label = label
        # setup the branch attribute
        assert node.parent.aname is None or node.parent.aname == aname
        node.branch_aname = aname
        if node.parent.aname is None:
            node.parent.aname = aname
        # setup the branch type
        assert atype in (DTNode.attr_num, DTNode.attr_sym)
        assert node.parent.atype is None or node.parent.atype == atype
        node.branch_atype = atype
        if node.parent.atype is None:
            node.parent.atype = atype
        # Store global attribute information in the root
        root = self._root
        if aname not in root._atypes:
            root._atypes[aname]= atype
            root._albs[aname] = -float('inf')
            root._aubs[aname] = float('inf')

    def set_class_label(self, class_label):
        # can be int (dtree) or float (rtree)
        self.class_label = eval(class_label)

    def eval(self, sample):
        # if the node is a leaf, just return the class
        if self.class_label is not None:
            return self.class_label
        # otherwise, "forward" the sample along one of the branches
        else:
            for child in self.children:
                # process branches for numeric attributes
                if self.atype == DTNode.attr_num:
                    vmin, vmax = child.branch_label
                    #print "vmin "+str(vmin)+", sample "+str(sample)+", max "+str(vma)
                    if not self.thr_left:
                        if vmin <= sample[self.aname] < vmax:
                            return child.eval(sample)
                    else:
                        if vmin < sample[self.aname] <= vmax:
                            return child.eval(sample)
                else:
                    msg = 'Symbolic attributes are not yet supported'
                    raise ValueError(msg)
            msg = 'A split on %s does not form a partition' % self.aname
            raise ValueError(msg)

    def __repr__(self):
        return _dt_to_string(self)

    def attributes(self):
        return self._root._atypes.keys()

    def lb(self, aname):
        return self._root._albs[aname]

    def update_lb(self, aname, value, tol=1e-4):
        albs = self._root._albs
        if albs[aname] < value-tol:
            albs[aname] = value
            return True
        else:
            return False

    def ub(self, aname):
        return self._root._aubs[aname]

    def update_ub(self, aname, value, tol=1e-4):
        aubs = self._root._aubs
        if aubs[aname] > value+tol:
            aubs[aname] = value
            return True
        else:
            return False

    def atype(self, atype):
        return self._root._atypes[aname]

    def reset_bounds(self):
        for aname in self._root._atypes:
            self._root._albs[aname] = -float('inf')
            self._root._aubs[aname] = float('inf')

    def nsplits(self):
        cnt = 0
        if self.branch_aname is not None:
            # not the (fake) root
            cnt += 1
        #if self.class_label != None:
        #    # a leaf, also count the class
        #    cnt += 1
        # process the children
        for child in self.children:
            cnt += child.size()
        return cnt


def _dt_to_string(tree, level=0, use_ref_lbl=False):
    # print level indicators
    res = '|   ' * (level-1)
    # ------------------------------------------------------------------------
    # print branch information
    if tree.branch_aname is not None:
        # print the attribute name
        res += 'attr_%d' % tree.branch_aname
        # print the split condition
        label = tree.branch_label if not use_ref_lbl else tree.refined_label
        if isinstance(label, tuple):
            if label[0] == -float('inf'):
                if not tree.thr_left:
                    res += ' < %f' % label[1]
                else:
                    res += ' <= %f' % label[1]
            else:
                if not tree.thr_left:
                    res += ' >= %f' % label[0]
                else:
                    res += ' > %f' % label[0]
        elif isinstance(label, list) or isinstance(label, set):
            res += ' in (' + ' '.join(str(v) for v in sorted(label)) + ')'
        else:
            msg = 'Unrecognized label type'
            raise ValueError(msg)
    # ------------------------------------------------------------------------
    # print the class information
    if tree.class_label != None:
        if not use_ref_lbl:
            res += (': %s\n' % tree.class_label)
        else:
            res += (': %d\n' % tree.refined_class_label)
    else:
        res += '\n' if level > 0 else ''
    # ------------------------------------------------------------------------
    # process the children
    for child in tree.children:
        res += _dt_to_string(child, level+1, use_ref_lbl)
    # ------------------------------------------------------------------------
    # return the string
    return res
