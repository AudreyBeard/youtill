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
        raise ValueError("{!s} is not divisible by {:d}".format(shape_a, dim_b1))  # NOQA

    if ndims == 3:
        out_shape = (1, dim_b1, int(dim_b2))
    else:
        out_shape = (dim_b1, int(dim_b2))

    if typecast is not None:
        return typecast(out_shape)
    else:
        return out_shape


def deprecated(f):
    def deprec_warn():
        deprecation_msg = '{} is deprecated - consider replacing it'.format(f.__name__)  # NOQA
        warn(deprecation_msg)
        f()
    return deprec_warn


if __name__ == "__main__":
    import doctest
    doctest.testmod()
