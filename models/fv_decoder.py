import tensorflow as tf
import numpy as np
import math
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
sys.path.append('/home/itzikbs/PycharmProjects/fisherpointnet/EMD')
import tf_auctionmatch
import tf_sampling




def placeholder_inputs(batch_size, n_points, gmm):

    #Placeholders for the data
    n_gaussians = gmm.means_.shape[0]
    D = gmm.means_.shape[1]

    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))

    w_pl = tf.placeholder(tf.float32, shape=(n_gaussians))
    mu_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D))
    sigma_pl = tf.placeholder(tf.float32, shape=(n_gaussians, D)) # diagonal
    points_pl = tf.placeholder(tf.float32, shape=(batch_size, n_points, D))

    return points_pl, labels_pl, w_pl, mu_pl, sigma_pl


def get_model(points, w, mu, sigma, is_training, bn_decay=None, weigth_decay=0.005, add_noise=False, num_classes=40):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = points.get_shape()[0].value
    n_points = points.get_shape()[1].value
    n_gaussians = w.shape[0].value
    res = int(np.round(np.power(n_gaussians,1.0/3.0)))


    # fv = tf_util.get_fv_minmax(points, w, mu, sigma, flatten=False)
    fv = tf_util.get_fv_tf(points, w, mu, sigma, flatten=False)

    grid_fisher = tf.reshape(fv,[batch_size,-1,res,res,res])
    grid_fisher = tf.transpose(grid_fisher, [0, 2, 3, 4, 1])

    #net = tf.reshape(grid_fisher,[batch_size, -1])

    #Decoder
    # Inception
    layer = 1
    net = inception_module(grid_fisher, n_filters=256, kernel_sizes=[3,5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    # layer = layer + 1
    # net = inception_module(net, n_filters=128,kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    # layer = layer + 1
    # net = inception_module(net, n_filters=256,kernel_sizes=[3, 5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    # layer = layer + 1
    # net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool'+str(layer), stride=[2, 2, 2], padding='SAME')
    # layer = layer + 1
    # net = inception_module(net, n_filters=256,kernel_sizes=[3,5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    # layer = layer + 1
    # net = inception_module(net, n_filters=512,kernel_sizes=[3,5], is_training=is_training, bn_decay=bn_decay, scope='inception'+str(layer))
    # layer = layer + 1
    # net = tf_util.max_pool3d(net, [2, 2, 2], scope='maxpool'+str(layer), stride=[2, 2, 2], padding='SAME')

    net = tf.reshape(net,[batch_size, -1])

    # net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
    #                               scope='fc'+str(layer), bn_decay=bn_decay, weigth_decay=weigth_decay)
    layer = layer+1
    net = tf_util.fully_connected(net, 1024, bn=True, is_training=is_training,
                                  scope='fc'+str(layer), bn_decay=bn_decay, weigth_decay=weigth_decay)
    layer = layer + 1
    net = tf_util.fully_connected(net, n_points*3, bn=True, is_training=is_training,
                                  scope='fc'+str(layer), bn_decay=bn_decay, weigth_decay=weigth_decay, activation_fn=None)

    reconstructed_points = tf.reshape(net,[batch_size, n_points, 3])


    return reconstructed_points, fv

def inception_module(input, n_filters=64, kernel_sizes=[3,5], is_training=None, bn_decay=None, scope='inception'):
    one_by_one =  tf_util.conv3d(input, n_filters, [1,1,1], scope= scope + '_conv1',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    three_by_three = tf_util.conv3d(one_by_one, int(n_filters/2), [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope= scope + '_conv2',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)
    five_by_five = tf_util.conv3d(one_by_one, int(n_filters/2), [kernel_sizes[1], kernel_sizes[1], kernel_sizes[1]], scope=scope + '_conv3',
                          stride=[1, 1, 1], padding='SAME', bn=True,
                          bn_decay=bn_decay, is_training=is_training)
    average_pooling = tf_util.avg_pool3d(input, [kernel_sizes[0], kernel_sizes[0], kernel_sizes[0]], scope=scope+'_avg_pool', stride=[1, 1, 1], padding='SAME')
    average_pooling = tf_util.conv3d(average_pooling, n_filters, [1,1,1], scope= scope + '_conv4',
           stride=[1, 1, 1], padding='SAME', bn=True,
           bn_decay=bn_decay, is_training=is_training)

    output = tf.concat([ one_by_one, three_by_three, five_by_five, average_pooling], axis=4)
    #output = output + tf.tile(input) ??? #resnet
    return output

def pairwise_diff(x, y):
        size_x = tf.shape(x)[1]
        size_y = tf.shape(y)[1]
        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, 1, size_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, 1, size_x]))
        yy = tf.transpose(yy, perm=[0, 3, 2, 1])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)
        square_dist = tf.reduce_sum(square_diff, axis=2)
        return square_dist

def get_loss(reconstructed_points, original_points, type='chamfer', w=0, mu=0, sigma=0,fv=0, add_fv_loss=False):

    n_points = reconstructed_points.get_shape()[1].value
    d = tf.constant(0)
    matched_out = tf.constant(0)

    if type == 'chamfer':
        #Chamfer Distance
        s1_s2 = tf.reduce_sum(tf.reduce_min(pairwise_diff(reconstructed_points, original_points), axis=2), axis=1)
        s2_s1 = tf.reduce_sum(tf.reduce_min(pairwise_diff(original_points, reconstructed_points), axis=2), axis=1)
        loss = (s1_s2 + s2_s1)/ n_points
    elif type == 'emd':
        matchl_out, matchr_out = tf_auctionmatch.auction_match(original_points, reconstructed_points)
        matched_out = tf_sampling.gather_point(reconstructed_points, matchl_out)
        d = tf.sqrt(tf.reduce_sum(tf.square(original_points - matched_out), axis=2))
        loss = tf.reduce_sum(d,axis=1)/ n_points
    elif type=='joint':
        #Use both loss functions
        s1_s2 = tf.reduce_sum(tf.reduce_min(pairwise_diff(reconstructed_points, original_points), axis=2), axis=1)
        s2_s1 = tf.reduce_sum(tf.reduce_min(pairwise_diff(original_points, reconstructed_points), axis=2), axis=1)
        loss_chamfer = s1_s2 + s2_s1

        matchl_out, matchr_out = tf_auctionmatch.auction_match(original_points, reconstructed_points)
        matched_out = tf_sampling.gather_point(reconstructed_points, matchl_out)
        d = tf.sqrt(tf.reduce_sum(tf.square(original_points - matched_out), axis=2))
        loss_emd = tf.reduce_sum(d,axis=1)
        loss = (loss_chamfer + loss_emd) / n_points

    loss = tf.reduce_mean(loss)
    tf.summary.scalar('emd_loss', loss)

    if add_fv_loss:
        fv_rec = tf_util.get_fv_tf_no_mvn(reconstructed_points, w, mu, sigma, flatten=False)
        fv_loss = tf.nn.l2_loss(fv_rec-fv)
        tf.summary.scalar('fv_loss', fv_loss)
        loss = loss + 0.001*fv_loss





    # tf.summary.scalar('weight_decay_loss', weight_decay_loss)
    return loss, d, matched_out

if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print (outputs)
