from numpy import prod
import collections
from warnings import warn

VERBOSITY = 0

operations = collections.OrderedDict()
operations['<='] = lambda x, y: x <= y
operations['>='] = lambda x, y: x >= y
operations['=='] = lambda x, y: x == y
operations['!='] = lambda x, y: x != y
operations['>'] = lambda x, y: x > y
operations['<'] = lambda x, y: x < y


def isiterable(item):
    return isinstance(item, collections.Iterable)


def isoperation(string):
    return string in operations.keys()


def shape_for_shape(shape_a, dim_b1, ndims=2, typecast=None):
    """ Given a tensor shape and a single dimension, computes the n-dimensional
    shape that satisfies the given single dimension, cast according to the
    typecast function

        Parameters:
            shape_a <list-like>: dimensionality of tensor
            dim_b1 <int>: single dimension of desired tensor shape
            ndims <int>: determines how many dimensions to expand into, if not 2
            typecast <function>: casts the output shape
        Returns:
            <type(typecast(<list-like>))> if typecast is given,
            <tuple> if typecast is None

        Example:
            >>> from numpy import arange
            >>> a = arange(1, 25).reshape(3, 8)
            >>> shape_for_shape(a.shape, 6)
            (6, 4)
            >>> shape_for_shape(a.shape, 1)
            (1, 24)
            >>> shape_for_shape(a.shape, 24)
            (24, 1)
            >>> shape_for_shape(a.shape, 12)
            (12, 2)
            >>> try:
            ...     shape_for_shape(a.shape, 5)
            ... except ValueError as err:
            ...     print("24 is not divisible by 5")
            24 is not divisible by 5

    """

    dim_b2 = prod(shape_a) / dim_b1
    if int(dim_b2) != dim_b2:
        raise ValueError("{!s} is not divisible by {:d}".format(shape_a, dim_b1))

    if ndims == 3:
        out_shape = (1, dim_b1, int(dim_b2))
    else:
        out_shape = (dim_b1, int(dim_b2))

    if typecast is not None:
        return typecast(out_shape)
    else:
        return out_shape


def check_constraints(param_value, param_name, constraints, check_all=True):
    """ function for checking if a parameter value is valid within constraints.
        Parameters:
            param_value: The value of the parameter to check
            param_name: The name of the parameter
            constraints: The dictionary containing all constraints on this
                parameter, including type or numerical constraints
        Returns:
            bool: whether the param_value is properly constrained
        Example:
            >>> v = dict(s=str,
            ...          n=('>=1', '<5'),
            ...          l=collections.Iterable,
            ...          na=(None),
            ...          e=(lambda x: True if x is None else isinstance(x, str)))
            >>> assert(check_constraints('a', 's', v))
            >>> assert(not check_constraints(1, 's', v))
            >>> assert(check_constraints(1, 'n', v))
            >>> assert(check_constraints(4, 'n', v))
            >>> assert(not check_constraints(5, 'n', v))
            >>> assert(check_constraints('anything', 'na', v))
            >>> assert(check_constraints('string', 'e', v))
            >>> assert(check_constraints(None, 'e', v))
            >>> assert(not check_constraints(1, 'e', v))
    """

    def _check_constraint(item, c, item_name=None):
        """ Helper to determine if item is correctly constrained by c
            Parameters:
                item: the value to check
                c: constraint to check against
        """

        def _isop(c, opstr):
            """ Helper to determine if a constraint is the same as an operation string
                Parameters:
                    c (str): constraint to check e.g. '<3', '>=0'
                    opstr (str): operation we're concerned with e.g. '<', '>='
                Returns:
                    (bool) True if constraint matches operation, False OTW
                Example:
                    >>> assert(_isop('<3', '<')
                    >>> assert(_isop('>=3', '>=')
                    >>> assert(_isop('==3', '==')
                    >>> assert(not _isop('==3', '!=')
                    >>> assert(not _isop('>=3', '>')
                    >>> assert(not _isop('<3', '<=')
            """
            return c.startswith(opstr) and not c.split(opstr)[1].startswith('=')

        def _getval(c, opstr):
            """ Helper to get the value on the RHS of the constraint - only
                called when (opstr in c) is True
                Parameters:
                    c (str): constraint containing operation e.g. '<3', '>=0'
                    opstr (str): operation we're pulling out e.g. '<', '>='
                Returns:
                    (float)
                Example:
                    >>> assert(_getval('<3', '<') == 3)
                    >>> assert(_getval('>=3', '>=') == 3)
                    >>> assert(_getval('==3', '==') == 3)
            """
            return float(c.split(opstr)[1])

        # No constraint
        if c is None:
            return True

        # Constrained by type
        elif type(c) == type or c == collections.Iterable:
            return isinstance(item, c)

        # Item should be a number and constrained by some operation
        elif isinstance(c, str):
            numeric = any([c.startswith(opstr) for opstr in operations.keys()])
            if numeric:
                try:
                    item = float(item)
                except (ValueError, TypeError):
                    return False
                else:
                    correct_or_dontcare = [op(item, _getval(c, opstr))        # Check operation
                                           if _isop(c, opstr)                 # Only if operation is constraint
                                           else True                          # OTW, don't care
                                           for opstr, op in operations.items()]  # For all operations given
                    return all(correct_or_dontcare)
            else:
                return item == c

        # If constraint is, for instance, a lambda function verifying that a
        # parameter is either None or satisfies some other constraint
        # e.g. (lambda x: x is None or isinstance(x, str))
        else:
            if VERBOSITY > 0:
                print('param {}\n  {}({}) evaluates to {}'.format(
                    item_name,
                    c.__name__,
                    item,
                    c(item) if c is not None else 'N/A'))
            return c(item)

    # First get the actual constraint(s)
    constraint = constraints.get(param_name)

    # If multiple constraints, check all (or any) of them
    if isiterable(constraint):
        if check_all:
            return all([_check_constraint(param_value, c, param_name) for c in constraint])
        else:
            return any([_check_constraint(param_value, c, param_name) for c in constraint])

    else:
        return _check_constraint(param_value, constraint, param_name)


class ParameterRegister(collections.OrderedDict):
    def __init__(self, constraints=None, defaults=None):
        super().__init__()
        self.constraints = constraints
        self.defaults = defaults

    def register(self, kwarg_name, constraints=None, default=None):
        self.constraints[kwarg_name] = constraints
        self.defaults[kwarg_name] = default

    def check_kwargs(self, **kwargs):
        valid = {k: check_constraints(v, k, self.constraints) for k, v in kwargs.items()}
        return valid

    def set_uninitialized_params(self, defaults=None):
        if defaults is not None:
            defaults_notset = {key: defaults[key]
                               for key in defaults
                               if self.get(key) is None}
        elif self.defaults is not None:
            defaults_notset = {key: self.defaults[key]
                               for key in self.defaults
                               if self.get(key) is None}
        else:
            print('No defaults dictionary given - cannot set')

        self.update(**defaults_notset)

    def _err_fmt(self, key, kwarg_dict):
        return '{} expects {} but got {}'.format(key, self.constraints[key], kwarg_dict[key])

    def set(self, **kwargs):
        no_check = kwargs.get('no_check')
        if not no_check:
            is_good = self.check_kwargs(**kwargs)
            if not all(is_good.values()):
                bad_kwargs = ', '.join([self._err_fmt(kwarg, kwargs) for kwarg, good in is_good.items() if not good])
                raise ValueError(bad_kwargs)

        for k, v in kwargs.items():
            self[k] = v

    # TODO fix this
    #def update(self, **kwargs):
    #    no_check = kwargs.get('no_check')
    #    if not no_check:
    #        is_good = self.check_kwargs(**kwargs)
    #        if not all(is_good.values()):
    #            bad_kwargs = ', '.join([kwarg for kwarg, good in is_good.items() if not good])
    #            raise ValueError(bad_kwargs)
    #    self._update(**kwargs)

    @property
    def hashable_str(self):
        return ','.join(['{}:{}'.format(k, v) for k, v in self.items()])

    @property
    def _pretty(self):
        import ipdb
        with ipdb.launch_ipdb_on_exception():
            s = self.__class__.__name__
            for k, v in self.items():
                s += '\n  {}: {}'.format(k, v.__repr__())
            s += '\nConstraints'
            for k, v in self.constraints.items():
                s += '\n  {}: {}'.format(k, v)
            s += '\nDefaults'
            for k, v in self.defaults.items():
                s += '\n  {}: {}'.format(k, v.__repr__())
            s += '\n'
        return s

    def __repr__(self):
        return self._pretty


def deprecated(f):
    def deprec_warn():
        deprecation_msg = '{} is deprecated - consider replacing it'.format(f.__name__)
        warn(deprecation_msg)
        f()
    return deprec_warn


if __name__ == "__main__":
    import doctest
    doctest.testmod()
