class DTNode(object):
    """ Class used to shape decision trees.

    This class provides the possibility of building a decision tree 
    and expanding it. In this project it will be used in order to port 
    decision trees build via scikit-learn (REF TODO).

    If the node has no attribute name, type, range and parent this means that
    the node is the root of a tree.

    Parameters
    ----------
        attr_name : string/int
            Name of the attribute,  (default None)
        attr_type : int
            Type of the attribute, either 0 (numeric) or 1 (symbolic), (default None)
        attr_range : (float, float)
            Defines the inteval of the split. A split is defined by two ranges
            one for the left and one for the right, (default None)
        _class : int
            Label of the class, if the current node is a leaf (default None)

    Attributes
    ----------
        _attr_name: string/int
            Name of the attribute
        _attr_type : int
            Type of the attribute, either 0 (numeric) or 1 (symbolic)
        _attr_range : (float, float)
            Defines the inteval of the split. A split is defined by two ranges
            one for the left and one for the right
        _class : int
            Label of the class, if the current node is a leaf
        _parent : :obj:`eml.tree.describe.DTNode`
        _children : list(:obj:`eml.tree.describe.DTNode`)
        thr_left : bool
            Defines the way the tree parses the intervals (TO BE REMOVED)

    Raises
    ------
        ValueError
            If the given range is not valid

    """
    def __init__(self, attr_name=None, attr_type=None,
                 attr_range=None, _class=None):
        self._attr_name = attr_name
        self._attr_type = attr_type
        # checking if the attribute is a valid interval
        if (attr_range is None) or (len(attr_range) == 2 and attr_range[0] < attr_range[1]):
            self._attr_range = attr_range
        else:
            raise ValueError('Only numeric attributes are supported',
                             'the attribute range must be a valid interval')
        self._parent = None
        self._children = []
        self._class = None
        # if a root is created 
        if self._attr_name is None:
            self.attributes_lb = {}
            self.attributes_ub = {}
        else:
            self.attributes_lb = None
            self.attributes_ub = None

    def add_child(self, child): 
        """ Add a child to the current node

        Updates the list children and adds lower
        and upper bounds for the child 
        if not already existing

        Parameters
        ----------
            child : :obj:`eml.tree.describe.DTNode`

        Returns
        -------
            child : :obj:`eml.tree.describe.DTNode`

        """
        # adding child to children of this node 
        self._children.append(child)
        # assign parent to child
        if child._parent is None:
            child._parent = self
        # updating lower and upper bounds
        if child.attr_name is not None:
            child.attributes_lb = self.attributes_lb
            child.attributes_ub = self.attributes_ub
            if child.attr_name() not in self.attributes_lb.keys():
                self.attributes_lb[child.attr_name()] = -float('inf')
                self.attributes_ub[child.attr_name()] = float('inf')
            
        return child

    def attr_name(self):
        """ Get the node's attribute

        Returns the name of the attribute to test
        in order to reach this node

        Returns
        -------
            Name : string/int
                Name of the attribute
        """
        return self._attr_name

    def attr_range(self):
        """ Get the range of the node's attribute

        Returns the range that the sample's attribute 
        must respect in order to reach this node

        Returns
        -------
            Interval : (float, float)
                Range of the attribute fo this node
        """
        return self._attr_range

    def attr_type(self):
        """ Get the type of the node's attribute

        Returns the type od the attribute to be tested
        in order to reach this node, it is either 0 
        (numeric) or 1 (symbolic)
        """
        return self._attr_type

    def get_class(self):
        """ Get the class label of the current node

        Returns 
        -------
            Class label : int 
                If this is a leaf an integer representing
                the class, None otherwise
        """
        return self._class

    def set_class(self, _class):
        """ Set the class label

        Parameters
        ----------
            _class : int
                Class label

        Returns
        -------
            None

        """
        self._class = _class

    def get_children(self):
        """ Get the list of the children of the node

        Returns
        -------
            Children : list(:obj:`eml.tree.describe.DTNode`)

        """
        return self._children

    def eval(self, sample):
        """ Evalueates a sample 

        Returns the category associated to the sample 
        in input according to the decision tree

        Parameters
        ----------
            sample : list(att : value)
                List of attribute-value 

        Returns 
        -------
            Class : int 
                Classification of the sample

        Raises
        ------
            ValueError
                If the sample is not well formed 

        """ 
        if self._class is not None:
            return self._class
        for child in self._children:
            if self._attr_type == 0:
                test_attribute = child.get_test_attribute()
                (tmin, tmax) = child.get_test_range()
                if tmin <= sample[test_attribute] < tmax:
                    return child.eval(sample)
        raise ValueError('Incorrect Sample')

    def lb(self, attr_name):
        """ Get lower bound of a attribute

        Parameters
        ----------
            attr_ame : int 
                Name of the attrbute of interest
        
        Returns
        -------
            Lower Bound : float
                Lower bound of the attribute specified in input 

        """
        return self.attributes_lb[attr_name]

    def update_lb(self, attr_name, value, tol=1e-4):
        """ Update lower bound of a attribute
        
        Parameters
        ----------
            attr_name : int
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
        albs = self.attributes_lb
        if albs[attr_name] < value-tol:
            self.attributes_lb[attr_name] = value
            return True
        else:
            return False

    def ub(self, attr_name):
        """ Get upper bound of a attribute

        Parameters
        ----------
            attr_name : int
                Name of the attrbute of interest
        
        Returns
        -------
            Upper Bound : float
                Upper bound of the attribute specified in input 

        """
        return self.attributes_ub[attr_name]

    def update_ub(self, attr_name, value, tol=1e-4):
        """ Update upper bound of a attribute
        
        Parameters
        ----------
            attr_name : int
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
        aubs = self.attributes_ub
        if aubs[attr_name] > value+tol:
            self.attributes_ub[attr_name] = value
            return True
        else:
            return False

    def __repr__(self):
        res = ""
        for child in self.get_children():
            res += _aux(child, 0, "")
        return res


def _aux(node, level, res):
    res = '|   ' * (level)
    class_ = node.get_class()
    res += 'attr_%s' % str(node.attr_name())
    range_ = node.attr_range()
    if range_[0] == -float('inf'):
        res += ' <= %f' % range_[1]
    else:
        res += ' > %f' % range_[0]
    if class_ is not None:  
        res += (': %s\n' % str(node.get_class()))
    else: 
        res += '\n' if level >= 0 else ''
    for child in node.get_children():
        res += _aux(child, level+1, res)
    return res
    
