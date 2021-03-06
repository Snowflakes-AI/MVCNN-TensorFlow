import numpy as np
import os,sys,inspect
import tensorflow as tf
import time
from datetime import datetime
import os
import hickle as hkl
import os.path as osp
from glob import glob
import sklearn.metrics as metrics
import math

from input import Dataset
import globals as g_

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import model

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', osp.dirname(sys.argv[0]) + '/tmp/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_string('weights', '',
                            """finetune with a pretrained model""")
tf.app.flags.DEFINE_string('log_pr_curve', '',
                            """path to write precision-recall curve""")

np.set_printoptions(precision=3)


def test(dataset, ckptfile):
    print('test() called')
    V = g_.NUM_VIEWS
    batch_size = FLAGS.batch_size

    data_size = dataset.size()
    print('dataset size:', data_size)

    with tf.Graph().as_default():
        startstep = 0
        global_step = tf.Variable(startstep, trainable=False)

        view_ = tf.placeholder('float32', shape=(None, V, g_.IMG_W, g_.IMG_H, 3), name='im0')
        y_ = tf.placeholder('int64', shape=(None), name='y')
        keep_prob_ = tf.placeholder('float32')

        fc8, aux_fcs = model.inference_multiview(view_, g_.NUM_CLASSES, keep_prob_)
        loss = model.loss(fc8, y_)
        if g_.VIEWPOOL == 'wavg':
            aux_losses = []
            num_per_grp = g_.NUM_VIEWS // g_.NUM_GROUPS
            for i in range(num_per_grp):
                if i == 0:
                    ys_ = y_
                else:
                    ys_ = tf.concat([ys_, y_], 0)

            for i, aux in enumerate(aux_fcs):
                grp_id = i // num_per_grp
                ind = i % num_per_grp
                if ind == 0:
                    fcs_per_grp = aux
                else:
                    fcs_per_grp = tf.concat([fcs_per_grp, aux], 0)

                if ind == (num_per_grp-1):
                    aux_losses.append(model.loss(fcs_per_grp, ys_, name='vg%02d_loss'%grp_id))
            aux_loss = tf.reduce_mean(aux_losses)
            loss = tf.add_n([loss, aux_loss])

        train_op = model.train(loss, global_step, data_size)
        prediction, confidence = model.classify(fc8)

        saver = tf.train.Saver(tf.all_variables(), max_to_keep=1000)

        init_op = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))

        saver.restore(sess, ckptfile)
        print('restore variables done')

        step = startstep

        predictions = []
        scores = []
        labels = []

        print("Start testing")
        print("Size:", data_size)
        print("It'll take", int(math.ceil(data_size/batch_size)), "iterations.")

        for batch_x, batch_y in dataset.batches(batch_size):
            step += 1

            start_time = time.time()
            feed_dict = {view_: batch_x,
                         y_ : batch_y,
                         keep_prob_: 1.0}

            pred, score, loss_value = sess.run(
                    [prediction,  confidence, loss,],
                    feed_dict=feed_dict)


            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                sec_per_batch = float(duration)
                print('%s: step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)' \
                     % (datetime.now(), step, loss_value,
                                FLAGS.batch_size/duration, sec_per_batch))

            predictions.extend(pred.tolist())
            scores.extend(score.tolist())
            labels.extend(batch_y.tolist())

        # print labels
        # print predictions
        acc = metrics.accuracy_score(labels, predictions)
        print('acc:', acc*100)

        # print precision & recall
        prcurve_fname = FLAGS.log_pr_curve.strip()
        if prcurve_fname:
            predictions = np.asarray(predictions)
            scores = np.asarray(scores)
            labels = np.asarray(labels)
            y_true = np.zeros_like(labels)
            y_true[np.where(predictions == labels)] = 1
            precision, recall, _ = metrics.precision_recall_curve(y_true, scores)
            np.savetxt(prcurve_fname, (precision, recall), delimiter=',')


def main(argv):
    st = time.time()
    print('start loading data')

    listfiles, labels = read_lists(g_.TEST_LOL)
    dataset = Dataset(listfiles, labels, subtract_mean=False, V=g_.NUM_VIEWS)

    print('done loading data, time=', time.time() - st)

    test(dataset, FLAGS.weights)


def read_lists(list_of_lists_file):
    listfile_labels = np.loadtxt(list_of_lists_file, dtype=str).tolist()
    listfiles, labels  = list(zip(*[(l[0], int(l[1])) for l in listfile_labels]))
    return listfiles, labels


if __name__ == '__main__':
    main(sys.argv)


