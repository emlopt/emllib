#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

# ===========================================================================
# Classes to model a decision tree
# ===========================================================================

class DTNode(object):
    """ Class used to shape decision trees.

    This class provides the possibility of building a decision tree 
    and expanding it. In this project it will be used in order to port 
    decision trees build via scikit-learn (REF TODO).

    Parameters
    ----------
        root : DTNode
            Root node (`default=None`)
    Attributes
    ----------
        _root : DTNode
            Root of the tree
        _atype : dict
            Type of the attributes in the tree
        _albs : dict
            Lower bounds of the attributes labeling the node 
        _aubs : dict
            Lower bounds of the attributes labeling the node 
        aname : 
            Name of attributes labeling the node 
        atype : int
            Type of the attributes labeling the node (just numeric now)
        parent : DTNode
            Parent node, equal to `self` if the node is root
        branch_label : 
            Label of the branch 
        branch_aname : 
            Branch attribute name
        branch_atype : 
            Branch attribute type
        children : List of DTNode
            Children of the current node 
        class_label : int 
            Label of the class, if the current node is a leaf

    """
    # attr_num, attr_sym = range(2)


    def __init__(self, root=None):
        self.attr_num = 0
        self.attr_sym = 1
        # Attribute type codes
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
        self.thr_left = True # falla secca 

    def add_child(self, node, aname, atype, label):
        """ Add one child to the node
        
        Parameters
        ----------
            node : DTNode
                Node to add in the children list
            aname : int 
                Name of the attribute leading from the current node 
                to the child is being added
            atype : int 
                Type of the attribute leading from the current node 
                to the child is being added
            label : int
                Label of the child node if it is a leaf

        """
        # TODO convert the assertions to exceptions
        # setup the parent-child relationship
        assert node.parent is None
        self.children.append(node)
        node.parent = self
        # setup the branch labels
        # assert not node.atype == DTNode.attr_num or len(label) == 2
        # assert not node.atype == DTNode.attr_num or label[0] < label[1]
        assert not node.atype == 0 or len(label) == 2
        assert not node.atype == 0 or label[0] < label[1]
        node.branch_label = label
        # setup the branch attribute
        assert node.parent.aname is None or node.parent.aname == aname
        node.branch_aname = aname
        if node.parent.aname is None:
            node.parent.aname = aname
        # setup the branch type
        # assert atype in (DTNode.attr_num, DTNode.attr_sym)
        assert atype in (0, 1)
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
        """ Set class label of the current node

        Parameters
        ----------
            class_label: int
                Integer representing the class of the node

        """
        # can be int (dtree) or float (rtree)
        self.class_label = eval(class_label)

    def eval(self, sample):
        """ Evaluates the output of the DT tree given a input.

        Given a sample each attribute of the input is tested according 
        to the thresholds of each node until a leaf node is reached.

        Parameters
        ----------
            sample:
                Input sample

        Returns
        ----------
            Class Label : int 
                Identifier of the class returned by the decision tree

        Raises
        ----------
            ValueError
                If there are symbolic attributes or inconsistent attributes 
                are tested

        """
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
                    msg = 'Symbolic attributes are not yet supported' # TODO
                    raise ValueError(msg)
            msg = 'A split on %s does not form a partition' % self.aname
            raise ValueError(msg)

    def __repr__(self):
        """ Representation of the DRNode

        Returns
        -------
            string:
                Representation of the tree/subtree

        """
        return _dt_to_string(self)

    def attributes(self):
        """ Get the list of the attributes in the tree

        Returns
        -------
            Attribute : list(int)
                List of the attributes currently in the tree

        """
        return self._root._atypes.keys()

    def lb(self, aname):
        """ Get lower bound of a attribute

        Parameters
        ----------
            aname : int 
                Name of the attrbute of interest
        
        Returns
        -------
            Lower Bound : float
                Lower bound of the attribute specified in input 

        """
        return self._root._albs[aname]

    def update_lb(self, aname, value, tol=1e-4):
        """ Update lower bound of a attribute
        
        Parameters
        ----------
            aname : int
                Name of the attrbute of interest
            value : float
                New lower bound value
            tol : float
                Tollerance
        
        Returns
        -------
            Acknowledge : bool
                True if the operation is succesfully exectued, 
                False otherwise

        """
        albs = self._root._albs
        if albs[aname] < value-tol:
            albs[aname] = value
            return True
        else:
            return False

    def ub(self, aname):
        """ Get upper bound of a attribute

        Parameters
        ----------
            aname : int
                Name of the attrbute of interest
        
        Returns
        -------
            Upper Bound : float
                Upper bound of the attribute specified in input 

        """
        return self._root._aubs[aname]

    def update_ub(self, aname, value, tol=1e-4):
        """ Update upper bound of a attribute
        
        Parameters
        ----------
            aname : int
                Name of the attrbute of interest
            value : float
                New upper bound value
            tol : float
                Tollerance
        
        Returns
        -------
            Acknowledge : bool
                True if the operation is succesfully exectued, 
                False otherwise

        """
        aubs = self._root._aubs
        if aubs[aname] > value+tol:
            aubs[aname] = value
            return True
        else:
            return False

    def atype(self, aname):
        """ Get type of a attribute

        Parameters
        ----------
            aname : int 
                Name of the attrbute of interest

        Returns
        ------- 
            Attribute Type : int 
                Type of the attribute specified 

        """
        return self._root._atypes[aname]

    def reset_bounds(self):
        """ Resets upper and lower bound for each attribute

        The lower and upper bound are setted respectively to 
        -inf and inf for each attribute in the tree.

        Returns
        -------
            None

        """
        for aname in self._root._atypes:
            self._root._albs[aname] = -float('inf')
            self._root._aubs[aname] = float('inf')

    def nsplits(self):
        """ Get number of splits in the tree.

        Returns
        -------
            Number of splits : int 
                Number of splits in the tree
        """
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
    """ Creates a string representing the tree

    TODO description representation

    Parameters
    ---------- 
        level : int 
            Starting level of the rappresentation. If 0 starts
            from the root.
        use_ref_lbl : bool
            Flag for refined class label in the representation

    Returns
    -------
        Representation : string 
            Representing the tree

    Raises
    ------
        ValueError
            If the label has a type not supported

    """ 
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
