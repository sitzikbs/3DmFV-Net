import os
import sys
import numpy as np

import matplotlib
matplotlib.use('pdf')
# import matplotlib.pyplot as plt
import importlib
import argparse
import tensorflow as tf
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_util
import visualization
import provider
import utils

# ModelNet40 official train/test split. MOdelNet10 requires separate downloading and sampling.
MAX_N_POINTS = 2048
NUM_CLASSES = 40
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet'+str(NUM_CLASSES)+'_ply_hdf5_'+ str(MAX_N_POINTS)+ '/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet'+str(NUM_CLASSES)+'_ply_hdf5_'+ str(MAX_N_POINTS)+ '/test_files.txt'))
LABEL_MAP = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet'+str(NUM_CLASSES)+'_ply_hdf5_'+ str(MAX_N_POINTS)+ '/shape_names.txt'))

print( "Loading Modelnet" + str(NUM_CLASSES))

#Execute
#python train_cls.py  --gpu=0 --log_dir='log' --batch_size=64 --num_point=1024 --num_gaussians=8 --gmm_variance=0.0156 --gmm_type='grid' --learning_rate=0.001  --model='voxnet_pfv' --max_epoch=200 --momentum=0.9 --optimizer='adam' --decay_step=200000  --weight_decay=0.0 --decay_rate=0.7

augment_rotation, augment_scale, augment_translation, augment_jitter, augment_outlier = (False, True, True, True, False)

parser = argparse.ArgumentParser()
#Parameters for learning
parser.add_argument('--gpu', type=int, default=2, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='3dmfv_net_cls', help='Model name [default: 3dmfv_net_cls]')
parser.add_argument('--log_dir', default='log_trial', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 64]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay coef [default: 0.0]')

# Parameters for GMM
parser.add_argument('--gmm_type',  default='grid', help='type of gmm [grid/learn], learn uses expectation maximization algorithm (EM) [default: grid]')
parser.add_argument('--num_gaussians', type=int , default=5, help='number of gaussians for gmm, if grid specify subdivisions, if learned specify actual number[default: 5, for grid it means 125 gaussians]')
parser.add_argument('--gmm_variance', type=float,  default=0.04, help='variance for grid gmm, relevant only for grid type')
FLAGS = parser.parse_args()


N_GAUSSIANS = FLAGS.num_gaussians
GMM_TYPE = FLAGS.gmm_type
GMM_VARIANCE = FLAGS.gmm_variance

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
WEIGHT_DECAY = FLAGS.weight_decay

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

#Creat log directory ant prevent over-write by creating numbered subdirectories
LOG_DIR = 'log/modelnet' + str(NUM_CLASSES) + '/' + FLAGS.model + '/'+ GMM_TYPE + str(N_GAUSSIANS) + '_' + FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
else:
    print('Log dir already exists! creating a new one..............')
    n = 0
    while True:
        n+=1
        new_log_dir = LOG_DIR+'/'+str(n)
        if not os.path.exists(new_log_dir):
            os.makedirs(new_log_dir)
            print('New log dir:'+new_log_dir)
            break
    FLAGS.log_dir = new_log_dir
    LOG_DIR = new_log_dir


os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_cls.py %s' % (LOG_DIR)) # bkp of train procedure
pickle.dump(FLAGS, open( os.path.join(LOG_DIR, 'parameters.p'), "wb" ) )

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
LOG_FOUT.write("augmentation RSTJ = " + str((augment_rotation, augment_scale, augment_translation, augment_jitter, augment_outlier))) #log augmentaitons

FAIL_CASES_FOUT = open(os.path.join(LOG_DIR, 'fail_cases.txt'), 'w')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

LIMIT_GPU = True

MAX_ACCURACY = 0.0
MAX_CLASS_ACCURACY = 0.0

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate


def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train(gmm):
    global MAX_ACCURACY, MAX_CLASS_ACCURACY
    # n_fv_features = 7 * len(gmm.weights_)

    # Build Graph, train and classify
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            points_pl, labels_pl, w_pl, mu_pl, sigma_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, gmm )
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, fv = MODEL.get_model(points_pl, w_pl, mu_pl, sigma_pl, is_training_pl, bn_decay=bn_decay, weigth_decay=WEIGHT_DECAY, add_noise=False, num_classes=NUM_CLASSES)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)
            # Get accuracy
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)#, aggregation_method = tf.AggregationMethod.EXPERIMENTAL_TREE) #consider using: tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        sess = tf_util.get_session(GPU_INDEX, limit_gpu=LIMIT_GPU)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl: True})

        ops = {'points_pl': points_pl,
               'labels_pl': labels_pl,
               'w_pl': w_pl,
               'mu_pl': mu_pl,
               'sigma_pl': sigma_pl,
               'is_training_pl': is_training_pl,
               'fv': fv,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(sess, ops, gmm, train_writer)
            acc, acc_avg_cls = eval_one_epoch(sess, ops, gmm, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            if acc > MAX_ACCURACY:
                MAX_ACCURACY = acc
                MAX_CLASS_ACCURACY = acc_avg_cls

        log_string("Best test accuracy: %f" % MAX_ACCURACY)
        log_string("Best test class accuracy: %f" % MAX_CLASS_ACCURACY)


def train_one_epoch(sess, ops, gmm, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    np.random.shuffle(train_file_idxs)

    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]], compensate = False)
        # points_idx = range(0,NUM_POINT)
        points_idx = np.random.choice(range(0,2048),NUM_POINT)
        current_data = current_data[:, points_idx, :]
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size / BATCH_SIZE

        total_correct = 0
        total_seen = 0
        loss_sum = 0

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            # Augment batched point clouds by rotation and jittering

            augmented_data = current_data[start_idx:end_idx, :, :]
            if augment_scale:
                augmented_data = provider.scale_point_cloud(augmented_data, smin=0.66, smax=1.5)
            if augment_rotation:
                augmented_data = provider.rotate_point_cloud(augmented_data)
            if augment_translation:
                augmented_data = provider.translate_point_cloud(augmented_data, tval = 0.2)
            if augment_jitter:
                augmented_data = provider.jitter_point_cloud(augmented_data, sigma=0.01,
                                                        clip=0.05)  # default sigma=0.01, clip=0.05
            if augment_outlier:
                augmented_data = provider.insert_outliers_to_point_cloud(augmented_data, outlier_ratio=0.02)

            feed_dict = {ops['points_pl']: augmented_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['w_pl']: gmm.weights_,
                         ops['mu_pl']: gmm.means_,
                         ops['sigma_pl']: np.sqrt(gmm.covariances_),
                         ops['is_training_pl']: is_training, }
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                             ops['train_op'], ops['loss'], ops['pred']],
                                                            feed_dict=feed_dict)

            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val

        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))


def eval_one_epoch(sess, ops, gmm, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    fail_cases_true_labels_final = []
    fail_cases_false_labes_final = []
    fail_cases_idx_final = []

    # points_idx = np.random.choice(range(0, 2048), NUM_POINT)
    points_idx = range(NUM_POINT)

    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn], compensate=False)
        current_data = current_data[:, points_idx, :]
        current_label = np.squeeze(current_label)

        file_size = current_data.shape[0]
        num_batches = file_size / BATCH_SIZE

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx + 1) * BATCH_SIZE

            feed_dict = {ops['points_pl']: current_data[start_idx:end_idx, :, :] ,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['w_pl']: gmm.weights_,
                         ops['mu_pl']: gmm.means_,
                         ops['sigma_pl']: np.sqrt(gmm.covariances_),
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])

            #Find the fail cases
            batch_current_label = current_label[start_idx:end_idx]
            false_idx = pred_val != batch_current_label
            fail_cases_true_labels = batch_current_label[np.where(false_idx)]  if batch_idx==0 else np.concatenate([fail_cases_true_labels,batch_current_label[np.where(false_idx)]] )
            fail_cases_false_labes = pred_val[np.where(false_idx)]  if batch_idx==0 else np.concatenate([fail_cases_false_labes, pred_val[np.where(false_idx)]])
            fail_cases_idx = false_idx if batch_idx == 0 else np.concatenate([fail_cases_idx, false_idx])

            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val * BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i - start_idx] == l)

        fail_cases_true_labels_final.append(fail_cases_true_labels)
        fail_cases_false_labes_final.append(fail_cases_false_labes)
        fail_cases_idx_final.append(fail_cases_idx)
    acc = total_correct / float(total_seen)
    acc_avg_cls =  np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (acc))
    log_string('eval avg class acc: %f' % (acc_avg_cls))

    FAIL_CASES_FOUT.write('True:' + str(fail_cases_true_labels) + '\n')
    FAIL_CASES_FOUT.write('Pred:' + str(fail_cases_false_labes) + '\n')
    FAIL_CASES_FOUT.write('Idx:' + str(fail_cases_idx) + '\n')
    FAIL_CASES_FOUT.flush()
    dump_dic = {'true_labels': fail_cases_true_labels_final,
                'false_pred_labels': fail_cases_false_labes_final,
                'idxs': fail_cases_idx_final}
    # pickle.dump([fail_cases_true_labels, fail_cases_false_labes], open(os.path.join(LOG_DIR, 'fail_cases.p'), "wb"))
    pickle.dump(dump_dic, open(os.path.join(LOG_DIR, 'fail_cases.p'), "wb"))

    return (acc, acc_avg_cls)


def export_visualizations(gmm, log_dir):
    """
    Visualizes and saves the images of the confusion matrix and fv representations

    :param gmm: instance of sklearn GaussianMixture (GMM) object Gauassian mixture model
    :param log_dir: path to the trained model
    :return None (exports images)
    """

    # load the model
    model_checkpoint = os.path.join(log_dir, "model.ckpt")
    if not(os.path.isfile(model_checkpoint+".meta")):
        raise ValueError("No log folder availabe with name " + str(log_dir))
    # reBuild Graph
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):

            points_pl, labels_pl,  w_pl, mu_pl, sigma_pl,  = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, gmm,)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Get model and loss
            pred, fv = MODEL.get_model(points_pl, w_pl, mu_pl, sigma_pl, is_training_pl, num_classes=NUM_CLASSES)

            ops = {'points_pl': points_pl,
                   'labels_pl': labels_pl,
                   'w_pl': w_pl,
                   'mu_pl': mu_pl,
                   'sigma_pl': sigma_pl,
                   'is_training_pl': is_training_pl,
                   'pred': pred,
                   'fv': fv}
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Create a session
            sess =  tf_util.get_session(GPU_INDEX, limit_gpu=LIMIT_GPU)

            # Restore variables from disk.
            saver.restore(sess, model_checkpoint)
            print("Model restored.")

            # Load the test data
            for fn in range(len(TEST_FILES)):
                log_string('----' + str(fn) + '-----')
                current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
                current_data = current_data[:, 0:NUM_POINT, :]
                current_label = np.squeeze(current_label)

                file_size = current_data.shape[0]
                num_batches = file_size / BATCH_SIZE

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx + 1) * BATCH_SIZE

                    feed_dict = {ops['points_pl']: current_data[start_idx:end_idx, :, :],
                                 ops['labels_pl']: current_label[start_idx:end_idx],
                                 ops['w_pl']: gmm.weights_,
                                 ops['mu_pl']: gmm.means_,
                                 ops['sigma_pl']: np.sqrt(gmm.covariances_),
                                 ops['is_training_pl']: False}

                    pred_label, fv_data = sess.run([ops['pred'], ops['fv']], feed_dict=feed_dict)
                    pred_label = np.argmax(pred_label, 1)

                    all_fv_data = fv_data if (fn==0 and batch_idx==0) else np.concatenate([all_fv_data, fv_data],axis=0)
                    true_labels = current_label[start_idx:end_idx] if (fn==0 and batch_idx==0) else np.concatenate([true_labels, current_label[start_idx:end_idx]],axis=0)
                    all_pred_labels = pred_label if (fn==0 and batch_idx==0) else np.concatenate([all_pred_labels, pred_label],axis=0)


    # Export Confusion Matrix
    visualization.visualize_confusion_matrix(true_labels, all_pred_labels, classes=LABEL_MAP, normalize=False, export=True,
                               display=False, filename=os.path.join(log_dir,'confusion_mat'), n_classes=NUM_CLASSES)

    # Export Fishre Vector Visualization
    label_tags = [LABEL_MAP[i] for i in true_labels]
    visualization.visualize_fv(all_fv_data, gmm, label_tags,  export=True,
                               display=False,filename=os.path.join(log_dir,'fisher_vectors'))
    # plt.show() #uncomment this to see the images in addition to saving them
    print("Confusion matrix and Fisher vectores were saved to /" + str(log_dir))


if __name__ == "__main__":

    gmm = utils.get_3d_grid_gmm(subdivisions=[N_GAUSSIANS, N_GAUSSIANS, N_GAUSSIANS], variance=GMM_VARIANCE)
    pickle.dump(gmm, open(os.path.join(LOG_DIR, 'gmm.p'), "wb"))
    train(gmm)
    #export_visualizations(gmm, LOG_DIR,n_model_limit=None)

    LOG_FOUT.close()


