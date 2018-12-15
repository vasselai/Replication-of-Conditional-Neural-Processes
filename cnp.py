"""
Conditional Neural Processes architecture
"""
import tensorflow as tf


def encoder(layersSizes, Xc, Yc, Nc):
    ''' MLP encoder h(.) in the paper, maps (x_c,y_c) -> r_c '''
    inputLayer = tf.concat([Xc, Yc], axis=-1)
    outputLayer = mlp(inputLayer, layersSizes, Nc, "encoder") 
    Rc = tf.reshape(outputLayer, (inputLayer.shape[0], Nc, layersSizes[-1]))
    return Rc


def aggregator(Rc):
    ''' Aggregates r_c for i..Nc into r_t, currently by a simple averaging '''
    r = tf.reduce_mean(Rc, axis=1)
    return r


def decoder(layersSizes, Xt, r, N):
    ''' MLP decoder g(.) in the paper, maps (x_t, r) -> mu_t, sigma_t '''
    r = tf.tile(tf.expand_dims(r, axis=1), [1, N, 1])
    inputLayer = tf.concat([r, Xt], axis=-1)
    outputLayer = mlp(inputLayer, layersSizes, N, "decoder")
    outputLayer = tf.reshape(outputLayer, (inputLayer.shape[0], N, -1))
    mu, sigma = tf.split(outputLayer, 2, axis=-1)
    sigma = tf.nn.softplus(sigma) #smooth approx. of ReLU for continuous outcomes
    sigma = 0.001 + 0.999 * sigma #to avoid collapsing (this was gotten from the original authors)
    return mu, sigma


def mlp(inputLayer, layersSizes, nPoints, modelName):
    """ Multi-Layer Perceptron with ReLU activation function in all layers
    but the last """
    #define layer structure:
    layers = tf.reshape(inputLayer, (inputLayer.shape[0] * nPoints, -1))
    layers.set_shape((None, inputLayer.shape[2]))
    #defines each hidden layer:
    for i, nNodes in enumerate(layersSizes[:-1]):
        layers = tf.layers.dense(layers, nNodes, activation=tf.nn.relu, \
                                 name = modelName + str(i),reuse=tf.AUTO_REUSE)
    #defines output layer:
    layers = tf.layers.dense(layers, layersSizes[-1], \
                             name = modelName + str(i+1),reuse=tf.AUTO_REUSE)
    return layers


def cnp(Xc, Yc, Nc, Xt, encoderLayersSizes, decoderLayersSizes):
    """ Executes the Conditional Neural Processes algorithm to learn
    Guassian Processes """
    #CNP's three processes - encoder, aggregator and decoder:
    Rc = encoder(encoderLayersSizes, Xc, Yc, tf.shape(Xc)[1])
    r = aggregator(Rc)
    mu, sigma = decoder(decoderLayersSizes, Xt, r, tf.shape(Xt)[1])
    return mu, sigma


