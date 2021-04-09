import tensorflow as tf
import os
from model import CPDNet

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("is_train", False, "if the train")
flags.DEFINE_boolean("matlab_bicubic", False, "using bicubic interpolation in matlab")
flags.DEFINE_integer("image_size", 64, "the size of cropped patch")
flags.DEFINE_integer("c_dim", 12, "the size of channel")
flags.DEFINE_integer("scale", 1, "the size of scale factor for preprocessing input image")
flags.DEFINE_integer("stride", 16, "the size of stride")
flags.DEFINE_integer("epoch", 20, "number of epoch")
flags.DEFINE_integer("batch_size", 16, "the size of batch")
flags.DEFINE_float("learning_rate", 1e-4 , "the learning rate")
flags.DEFINE_float("lr_decay_steps", 10 , "steps of learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.5 , "rate of learning rate decay")
flags.DEFINE_string("test_img", "", "test_img")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "name of the checkpoint directory")
flags.DEFINE_string("result_dir", "result", "name of the result directory")
flags.DEFINE_string("train_set_dir", "OL_CPDNET_DATA/Train", "dir of the train set")
flags.DEFINE_string("test_set_dir", "OL_CPDNET_DATA/Test", "dir of the test set")
flags.DEFINE_string("train_set", "train_dataset", "name of the train set")
flags.DEFINE_string("test_set", "test_dataset", "name of the test set")
flags.DEFINE_integer("D", 16, "D")
flags.DEFINE_integer("C", 8, "C")
flags.DEFINE_integer("G", 64, "G")
flags.DEFINE_integer("G0", 128, "G0")
flags.DEFINE_integer("kernel_size", 3, "the size of kernel")


def main(_):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cpdnet = CPDNet(tf.Session(),
                    is_train = FLAGS.is_train,
                    image_size = FLAGS.image_size,
                    c_dim = FLAGS.c_dim,
                    scale = FLAGS.scale,
                    batch_size = FLAGS.batch_size,
                    D = FLAGS.D,
                    C = FLAGS.C,
                    G = FLAGS.G,
                    G0 = FLAGS.G0,
                    kernel_size = FLAGS.kernel_size,
                    train_set_dir = FLAGS.train_set_dir,
                    test_set_dir=FLAGS.test_set_dir
              )

    if cpdnet.is_train:
        cpdnet.train(FLAGS)
    else:
        cpdnet.test(FLAGS)


if __name__=='__main__':
    tf.app.run()
