import tensorflow as tf
import numpy as np
from collections import namedtuple
from misc import *


def qr_loss(y, f, tau, sample_weights=None):
    err = y - f
    rho = tf.reduce_sum(tf.maximum(tau*err, (tau-1)*err), axis=1, keepdims=True)

    if sample_weights is None:
        return tf.reduce_mean(rho, axis=None)
    else:
        return tf.reduce_mean(
            tf.matmul(tf.constant(np.diag(sample_weights), dtype=tf.float32), rho),
            axis=None)


def predict_mvcaviar(x_train, prev_quantile, pars):
    tmax, n = np.shape(x_train)
    y_predict = np.matmul(x_train, pars['kernel']) + np.matmul(np.ones((tmax, 1)), np.reshape(pars['bias'], (1, n)))
    y_predict[0, :] += np.matmul(np.reshape(prev_quantile, (1, 2)),  pars['recurrent'])[0, :]
    for t in range(1, tmax):
        y_predict[t, :] += np.matmul(np.reshape(y_predict[t-1, :], (1, 2)), pars['recurrent'])[0, :]

    return y_predict


Result = namedtuple("Result", ["y_predict", "loss", "pars", "prev_quantile"])


def train_mvcaviar(Y, tau, init_pars=None, prev_quantile=None,
                   sample_weights=None, epochs=2000, verbose=False):
    tf.reset_default_graph()

    (tmax, n) = np.shape(Y)
    assert (tmax > 1)

    x_train = np.abs(Y[:-1, :])
    x_train = np.reshape(x_train, (tmax - 1, n))
    y_train = np.reshape(Y[1:, :], (tmax - 1, n))

    inputs = tf.placeholder(tf.float32, shape=(tmax-1, n))
    answers = tf.placeholder(tf.float32, shape=(tmax-1, n))

    param_names = ['kernel', 'recurrent', 'bias']
    if init_pars is None:
        initializers = {name: tf.initializers.zeros() for name in param_names}
    else:
        initializers = {name: tf.constant_initializer(init_pars[name]) for name in param_names}

    shapes = [(n, n), (n, n), (1, n)]
    params = {name: tf.get_variable(name, shape, initializer=initializers[name])
              for name, shape in zip(param_names, shapes)}

    if prev_quantile is None:
        prevq = tf.get_variable("prev_quantile", (1, n),
                                initializer=tf.initializers.zeros())
    else:
        prevq = tf.constant(np.reshape(prev_quantile, (1, n)), dtype=tf.float32)

    quantiles = [prevq]
    for t in range(0, tmax-1):
        a = tf.matmul(quantiles[t], params['recurrent']) + \
            tf.matmul(inputs[t : (t+1), :], params['kernel']) + params['bias']
        quantiles.append(a)

    outputs = tf.concat(quantiles[1:], axis=0)
    loss = qr_loss(answers, outputs, tau, sample_weights=sample_weights)

    opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.05)
    opt_operation = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(epochs):
            # Do gradient descent step
            _, loss_val = sess.run([opt_operation, loss], feed_dict={inputs: x_train, answers: y_train})
            if verbose:
                print("Epoch {}/{}: {}".format(j, epochs, loss_val))

        y_predict = sess.run(outputs, feed_dict={inputs: x_train})

        pars = {}
        for name in param_names:
            pars[name] = sess.run(params[name])

        prevq_val = sess.run(prevq)
        loss_val = sess.run(loss, feed_dict={inputs: x_train, answers: y_train})

    tf.reset_default_graph()
    return Result(y_predict, loss_val, pars, prevq_val)


def train_mvcaviar_separated(Y, tau, brek,
                             init_pars=None, prev_quantile=None,
                             sample_weights=None, epochs=2000, verbose=False):
    init_pars_left = get_left_pars(init_pars)
    sample_weights_left = None if sample_weights is None else sample_weights[:brek-1]
    res_left = train_mvcaviar(Y[:brek, :], tau,
                              prev_quantile=prev_quantile, init_pars=init_pars_left,
                              sample_weights=sample_weights_left, epochs=epochs, verbose=verbose)
    init_pars_right = get_right_pars(init_pars)
    sample_weights_right = None if sample_weights is None else sample_weights[brek-1:]
    res_right = train_mvcaviar(Y[brek-1:, :], tau,
                               prev_quantile=res_left.y_predict[-1, :], init_pars=init_pars_right,
                               sample_weights=sample_weights_right, epochs=epochs, verbose=verbose)
    pars_all = pars_left_right(res_left.pars, res_right.pars)
    y_predict = np.concatenate((res_left.y_predict, res_right.y_predict), axis=0)

    tmax = np.shape(Y)[0]
    loss = (res_left.loss * (brek-1) + res_right.loss * (tmax - brek)) / (tmax - 1)

    return Result(y_predict, loss, pars_all, res_left.prev_quantile)


def train_mvcaviar_w_shift(Y, tau, brek, shift,
                           init_pars=None, prev_quantile=None,
                           sample_weights=None, epochs=2000, verbose=False):
    tf.reset_default_graph()

    (tmax, n) = np.shape(Y)
    assert (tmax > 1)

    x_train = np.abs(Y[:-1, :])
    x_train = np.reshape(x_train, (tmax - 1, n))
    y_train = np.reshape(Y[1:, :], (tmax - 1, n))

    inputs = tf.placeholder(tf.float32, shape=(tmax - 1, n))
    answers = tf.placeholder(tf.float32, shape=(tmax - 1, n))

    param_names = ['kernel', 'recurrent', 'bias']

    initializers = {}
    if init_pars is None:
        for name in param_names:
            initializers[name] = tf.initializers.zeros()
    else:
        for name in param_names:
            initializers[name] = tf.constant_initializer(init_pars[name])

    params = {}
    shapes = [(n, n), (n, n), (1, n)]
    for name, shape in zip(param_names, shapes):
        params[name] = tf.get_variable(name, shape,
                                       initializer=initializers[name])

    if prev_quantile is None:
        prevq = tf.get_variable("prev_quantile", (1, n),
                                initializer=tf.initializers.zeros())
    else:
        prevq = tf.constant(np.reshape(prev_quantile, (1, n)), dtype=tf.float32)

    shift_const = {}
    for name in param_names:
        shift_const[name] = tf.constant(shift[name], dtype=tf.float32)

    quantiles = [prevq]
    for t in range(0, tmax - 1):
        a = tf.matmul(quantiles[t], params['recurrent']) + \
            tf.matmul(inputs[t: (t + 1), :], params['kernel']) + params['bias']
        if t >= brek:
            a = a + tf.matmul(quantiles[t], shift_const['recurrent']) + \
                tf.matmul(inputs[t: (t + 1), :], shift_const['kernel']) + shift_const['bias']
        quantiles.append(a)

    outputs = tf.concat(quantiles[1:], axis=0)
    loss = qr_loss(answers, outputs, tau, sample_weights=sample_weights)

    #err = answers - outputs
    #loss = tf.reduce_mean(tf.maximum(tau * err, (tau - 1) * err))

    opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.05)
    opt_operation = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(epochs):
            # Do gradient descent step
            _, loss_val = sess.run([opt_operation, loss], feed_dict={inputs: x_train, answers: y_train})
            if verbose:
                print("Epoch {}/{}: {}".format(j, epochs, loss_val))

        y_predict = sess.run(outputs, feed_dict={inputs: x_train})

        pars = {}
        for name in param_names:
            pars[name] = sess.run(params[name])

        prevq_val = sess.run(prevq)
        loss_val = sess.run(loss, feed_dict={inputs: x_train, answers: y_train})

    tf.reset_default_graph()
    return Result(y_predict, loss_val, pars, prevq_val)


def train_mvcaviar_separated2(Y, tau, brek,
                             init_pars=None, prev_quantile=None,
                             sample_weights=None, epochs=2000, verbose=False):
    tf.reset_default_graph()

    (tmax, n) = np.shape(Y)
    assert (tmax > 1)

    x_train = np.abs(Y[:-1, :])
    x_train = np.reshape(x_train, (tmax - 1, n))
    y_train = np.reshape(Y[1:, :], (tmax - 1, n))

    inputs = tf.placeholder(tf.float32, shape=(tmax - 1, n))
    answers = tf.placeholder(tf.float32, shape=(tmax - 1, n))

    param_names = ['kernel_left', 'recurrent_left', 'bias_left',
                   'kernel_right', 'recurrent_right', 'bias_right']

    initializers = {}
    if init_pars is None:
        for name in param_names:
            initializers[name] = tf.initializers.zeros()
    else:
        for name in param_names:
            initializers[name] = tf.constant_initializer(init_pars[name])

    params = {}
    shapes = [(n, n), (n, n), (1, n),
              (n, n), (n, n), (1, n)]
    for name, shape in zip(param_names, shapes):
        params[name] = tf.get_variable(name, shape,
                                       initializer=initializers[name])

    if prev_quantile is None:
        prevq = tf.get_variable("prev_quantile", (1, n),
                                initializer=tf.initializers.zeros())
    else:
        prevq = tf.constant(np.reshape(prev_quantile, (1, n)), dtype=tf.float32)


    quantiles = [prevq]
    for t in range(0, tmax - 1):
        if t < brek:
            a = tf.matmul(quantiles[t], params['recurrent_left']) + \
            tf.matmul(inputs[t: (t + 1), :], params['kernel_left']) + params['bias_left']
        if t >= brek:
            a = tf.matmul(quantiles[t], params['recurrent_right']) + \
                tf.matmul(inputs[t: (t + 1), :], params['kernel_right']) + params['bias_right']
        quantiles.append(a)

    outputs = tf.concat(quantiles[1:], axis=0)
    loss = qr_loss(answers, outputs, tau, sample_weights=sample_weights)

    opt = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.05)
    opt_operation = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for j in range(epochs):
            # Do gradient descent step
            _, loss_val = sess.run([opt_operation, loss], feed_dict={inputs: x_train, answers: y_train})
            if verbose:
                print("Epoch {}/{}: {}".format(j, epochs, loss_val))

        y_predict = sess.run(outputs, feed_dict={inputs: x_train})

        pars = {}
        for name in param_names:
            pars[name] = sess.run(params[name])

        prevq_val = sess.run(prevq)
        loss_val = sess.run(loss, feed_dict={inputs: x_train, answers: y_train})

    tf.reset_default_graph()
    return Result(y_predict, loss_val, pars, prevq_val)
