import numpy as np
import theano
from theano import tensor as T


def do_simple_theano_calculations():
    x1 = T.scalar()
    w1 = T.scalar()
    w0 = T.scalar()
    z1 = w1 * x1 + w0
    net_input = theano.function(inputs=[w1, x1, w0], outputs=z1)
    print(net_input(2.0, 1.0, 0.5))
    print()

    x = T.fmatrix(name='x')
    x_sum = T.sum(x, axis=0)
    calc_sum = theano.function(inputs=[x], outputs=x_sum)
    arr = [[1, 2, 3], [1, 2, 3]]
    print("Column sum: %s" % calc_sum(arr))
    arr = np.array(arr, dtype=theano.config.floatX)
    print("Column sum: %s" % calc_sum(arr))
    print()

    x = T.fmatrix(name='x')
    w = theano.shared(np.asarray(
        [[0.0, 0.0, 0.0]],
        dtype=theano.config.floatX,
    ))
    z = x.dot(w.T)
    update = [[w, w + 1.0]]
    net_input = theano.function(inputs=[x], updates=update, outputs=z)
    data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
    for i in range(5):
        print("z%d: %s" % (i, net_input(data)))
    print()

    data = np.array([[1, 2, 3]], dtype=theano.config.floatX)
    x = T.fmatrix(name='x')
    w = theano.shared(np.asarray(
        [[0.0, 0.0, 0.0]],
        dtype=theano.config.floatX,
    ))
    z = x.dot(w.T)
    update = [[w, w + 1.0]]
    net_input = theano.function(
        inputs=[],
        updates=update,
        givens={x: data},
        outputs=z,
    )
    for i in range(5):
        print("z%d: %s" % (i, net_input()))


if __name__ == '__main__':
    do_simple_theano_calculations()
