import tensorflow as tf
import numpy as np
import cv2
import time
import os
from utils import (
    input_setup,
    get_data_dir,
    get_data_num,
    get_batch,
    get_image,
    checkimage,
    imsave,
    tf_keepGradient_loss,
    #imread,
    prepare_data,
    PSNR,
    modcrop
)


class CPDNet(object):
    
    def __init__(self,
                 sess,
                 is_train,
                 image_size,
                 c_dim,
                 scale,
                 batch_size,
                 D,
                 C,
                 G,
                 G0,
                 kernel_size,
                 train_set_dir,
                 test_set_dir
                 ):

        self.sess = sess
        self.is_train = is_train
        self.image_size = image_size
        self.c_dim = c_dim
        self.scale = scale
        self.batch_size = batch_size
        self.D = D
        self.C = C
        self.G = G
        self.G0 = G0
        self.kernel_size = kernel_size
        self.train_set_dir = train_set_dir
        self.test_set_dir = test_set_dir

    def model(self, is_train):

        def _conv_unit(bottom, channel_size, unit_name, trainable=True):
            conv_feature = tf.layers.conv2d(bottom, channel_size, [3, 3],
                                            1, 'same', trainable=trainable, name=unit_name + '_Conv1')
            norm_feature = tf.layers.batch_normalization(conv_feature, trainable=trainable, name=unit_name + '_norm1')
            relu_feature = tf.nn.relu(norm_feature)

            return relu_feature

        def _resnet_unit(bottom, channel_list, unit_name, trainable=True):
            block_conv_0 = tf.layers.conv2d(bottom, channel_list[2], [1, 1],
                                            1, 'same', trainable=trainable, name=unit_name + '_lovalConv0')
            block_conv_1 = tf.layers.conv2d(block_conv_0, channel_list[0], [1, 1],
                                            1, 'same', trainable=trainable, name=unit_name+'_lovalConv1')
            block_norm_1 = tf.layers.batch_normalization(block_conv_1, trainable=trainable, name=unit_name+'_block_norm_1')
            block_relu_1 = tf.nn.relu(block_norm_1)

            block_conv_2 = tf.layers.conv2d(block_relu_1, channel_list[1], [3, 3],
                                            1, 'same', trainable=trainable, name=unit_name+'_lovalConv2')
            block_norm_2 = tf.layers.batch_normalization(block_conv_2, trainable=trainable, name=unit_name+'_block_norm_2')
            block_relu_2 = tf.nn.relu(block_norm_2)

            block_conv_3 = tf.layers.conv2d(block_relu_2, channel_list[2], [1, 1],
                                            1, 'same', trainable=trainable, name=unit_name+'_lovalConv3')
            block_norm_3 = tf.layers.batch_normalization(block_conv_3, trainable=trainable, name=unit_name+'_block_norm_3')
            block_res = tf.add(block_conv_0, block_norm_3)
            relu = tf.nn.relu(block_res)

            return relu

        unit1 = tf.layers.conv2d(self.images, 3, [1, 1], 1, 'same', trainable=is_train, name='unit1')
        unit2 = tf.layers.conv2d(unit1, 64, [3, 3], 1, 'same', trainable=is_train, name='unit2')
        unit3 = _resnet_unit(unit2, [self.D, self.C, self.G], 'unit3', is_train)

        unit_rgb = tf.layers.conv2d(unit3, 3, [3, 3], 1, 'same', trainable=is_train, name='unit_rgb')
        unit_r, unit_g, unit_b = tf.split(unit_rgb, 3, 3)

        unit_r_p_1 = tf.layers.conv2d(unit_r, 4, [3, 3], 1, 'same', trainable=is_train, name='unit_r_p_1')
        unit_r_p_2 = tf.layers.conv2d(unit_r_p_1, 64, [3, 3], 1, 'same', trainable=is_train, name='unit_r_p_2')
        unit_r_p_3 = _resnet_unit(unit_r_p_2, [self.D, self.C, self.G], 'unit_r_p_3', is_train)
        unit_r_p = tf.layers.conv2d(unit_r_p_3, 4, [3, 3], 1, 'same', trainable=is_train, name='unit_r_p')

        unit_g_p_1 = tf.layers.conv2d(unit_g, 4, [3, 3], 1, 'same', trainable=is_train, name='unit_g_p_1')
        unit_g_p_2 = tf.layers.conv2d(unit_g_p_1, 64, [3, 3], 1, 'same', trainable=is_train, name='unit_g_p_2')
        unit_g_p_3 = _resnet_unit(unit_g_p_2, [self.D, self.C, self.G], 'unit_g_p_3', is_train)
        unit_g_p = tf.layers.conv2d(unit_g_p_3, 4, [3, 3], 1, 'same', trainable=is_train, name='unit_g_p')

        unit_b_p_1 = tf.layers.conv2d(unit_b, 4, [3, 3], 1, 'same', trainable=is_train, name='unit_b_p_1')
        unit_b_p_2 = tf.layers.conv2d(unit_b_p_1, 64, [3, 3], 1, 'same', trainable=is_train, name='unit_b_p_2')
        unit_b_p_3 = _resnet_unit(unit_b_p_2, [self.D, self.C, self.G], 'unit_b_p_3', is_train)
        unit_b_p = tf.layers.conv2d(unit_b_p_3, 4, [3, 3], 1, 'same', trainable=is_train, name='unit_b_p')

        merge = tf.concat([unit3, unit_rgb, unit_r_p, unit_g_p, unit_b_p], 3, name='concatenate_7')
        unit4 = _conv_unit(merge, 256, 'unit4', is_train)
        unit5 = _conv_unit(unit4, 128, 'unit5', is_train)

        output = tf.layers.conv2d(unit5, 12, [1, 1], 1, 'same', trainable=is_train, name='output')

        return unit_rgb, unit_r_p, unit_g_p, unit_b_p, output

    def build_model(self, images_shape, labels_shape, labels_rgb_shape,
                    labels_rp_shape, labels_gp_shape, labels_bp_shape, is_train):

        self.images = tf.placeholder(tf.float32, images_shape, name='images')
        self.labels = tf.placeholder(tf.float32, labels_shape, name='labels')
        self.rgb_labels = tf.placeholder(tf.float32, labels_rgb_shape, name='rgb_labels')
        self.rp_labels = tf.placeholder(tf.float32, labels_rp_shape, name='rp_labels')     
        self.gp_labels = tf.placeholder(tf.float32, labels_gp_shape, name='gp_labels')
        self.bp_labels = tf.placeholder(tf.float32, labels_bp_shape, name='bp_labels')
        
        self.unit_rgb, self.unit_r_p, self.unit_g_p, self.unit_b_p, self.pred = self.model(self.is_train)      

        self.l1_0 = tf.losses.absolute_difference(self.labels[:, :, :, 0:3], self.pred[:, :, :, 0:3])    
        self.l1_45 = tf.losses.absolute_difference(self.labels[:, :, :, 3:6], self.pred[:, :, :, 3:6])  
        self.l1_90 = tf.losses.absolute_difference(self.labels[:, :, :, 6:9], self.pred[:, :, :, 6:9])  
        self.l1_135 = tf.losses.absolute_difference(self.labels[:, :, :, 9:12], self.pred[:, :, :,9:12])  
        self.mse_loss = tf.reduce_sum(self.l1_0 + self.l1_45 + self.l1_90 + self.l1_135)
                
        self.ssim_0 = tf.reduce_mean ([1. - tf.image.ssim (self.labels[k, :, :, 0:3], self.pred[k, :, :, 0:3], max_val=1.)
                                  for k in range (self.batch_size)])
        self.ssim_45 = tf.reduce_mean ([1. - tf.image.ssim (self.labels[k, :, :, 3:6], self.pred[k, :, :, 3:6], max_val=1.)
                                  for k in range (self.batch_size)])
        self.ssim_90 = tf.reduce_mean ([1. - tf.image.ssim (self.labels[k, :, :, 6:9], self.pred[k, :, :, 6:9], max_val=1.)
                                  for k in range (self.batch_size)])
        self.ssim_135 = tf.reduce_mean ([1. - tf.image.ssim (self.labels[k, :, :, 9:12], self.pred[k, :, :, 9:12], max_val=1.)
                                  for k in range (self.batch_size)])
        self.ssim_loss = tf.reduce_sum(self.ssim_0 + self.ssim_45 + self.ssim_90 + self.ssim_135)

        self.gra_0 = tf_keepGradient_loss(self.labels[:, :, :, 0:3], self.pred[:, :, :, 0:3])    
        self.gra_45 = tf_keepGradient_loss(self.labels[:, :, :, 3:6], self.pred[:, :, :, 3:6])  
        self.gra_90 = tf_keepGradient_loss(self.labels[:, :, :, 6:9], self.pred[:, :, :, 6:9])  
        self.gra_135 = tf_keepGradient_loss(self.labels[:, :, :, 9:12], self.pred[:, :, :,9:12])  
        self.gra_loss = tf.reduce_sum(self.gra_0 + self.gra_45 + self.gra_90 + self.gra_135)
        
        conv_w = tf.Variable(initial_value=tf.constant(
                        [[[[ 0.5,  1.,   0. ],
                           [ 0.5,  0.,   1. ],
                           [ 0.5, -1.,   0. ],
                           [ 0.5,  0.,  -1. ]]]]),trainable=False)
        
        self.gp_label = tf.stack([self.labels[:, :, :, 0], self.labels[:, :, :, 3], self.labels[:, :, :, 6], self.labels[:, :, :, 9]], 3)
        self.bp_label = tf.stack([self.labels[:, :, :, 1], self.labels[:, :, :, 4], self.labels[:, :, :, 7], self.labels[:, :, :, 10]], 3)
        self.rp_label = tf.stack([self.labels[:, :, :, 2], self.labels[:, :, :, 5], self.labels[:, :, :, 8], self.labels[:, :, :, 11]], 3) 
        self.output_r_Stokes_labels = tf.nn.conv2d(self.rp_label,conv_w,strides=[1,1,1,1],padding='VALID')
        self.output_g_Stokes_labels = tf.nn.conv2d(self.gp_label,conv_w,strides=[1,1,1,1],padding='VALID')
        self.output_b_Stokes_labels = tf.nn.conv2d(self.bp_label,conv_w,strides=[1,1,1,1],padding='VALID')
        
        self.gp_pred = tf.stack([self.pred[:, :, :, 0], self.pred[:, :, :, 3], self.pred[:, :, :, 6], self.pred[:, :, :, 9]], 3)
        self.bp_pred = tf.stack([self.pred[:, :, :, 1], self.pred[:, :, :, 4], self.pred[:, :, :, 7], self.pred[:, :, :, 10]], 3)
        self.rp_pred = tf.stack([self.pred[:, :, :, 2], self.pred[:, :, :, 5], self.pred[:, :, :, 8], self.pred[:, :, :, 11]], 3) 
        self.output_r_Stokes_pred = tf.nn.conv2d(self.rp_pred,conv_w,strides=[1,1,1,1],padding='VALID')
        self.output_g_Stokes_pred = tf.nn.conv2d(self.gp_pred,conv_w,strides=[1,1,1,1],padding='VALID')
        self.output_b_Stokes_pred = tf.nn.conv2d(self.gp_pred,conv_w,strides=[1,1,1,1],padding='VALID')
        
        self.loss_r_Stokes = tf.losses.absolute_difference(self.output_r_Stokes_labels, self.output_r_Stokes_pred)
        self.loss_g_Stokes = tf.losses.absolute_difference(self.output_g_Stokes_labels, self.output_g_Stokes_pred)
        self.loss_b_Stokes = tf.losses.absolute_difference(self.output_b_Stokes_labels, self.output_b_Stokes_pred)
        self.Stokes_loss =tf.reduce_sum(self.loss_r_Stokes + self.loss_g_Stokes + self.loss_b_Stokes)
        
        self.loss = (1-0.16)*self.mse_loss + 0.16*self.ssim_loss + 5*self.gra_loss + self.Stokes_loss
        self.total_error = tf.losses.absolute_difference(self.pred, self.labels)

        self.summary = tf.summary.scalar('accuracy', self.total_error )
        #self.summary = tf.summary.scalar('loss', self.loss)

        self.model_name = "OL_final"
        self.saver = tf.train.Saver(max_to_keep=1)

    def train(self, config):

        print("\nPrepare Data...\n")
        data = input_setup(config)
        if len(data) == 0:
            print("\nCan Not Find Training Data!\n")
            return

        data_dir = get_data_dir(config.checkpoint_dir, config.is_train)
        data_num = get_data_num(data_dir)
        batch_num = data_num // config.batch_size

        images_shape = [None, self.image_size, self.image_size, 1]
        labels_shape = [None, self.image_size, self.image_size, self.c_dim]
        labels_rgb_shape = [None, self.image_size, self.image_size, 3]
        labels_rp_shape = [None, self.image_size, self.image_size, 4]
        labels_gp_shape = [None, self.image_size, self.image_size, 4]
        labels_bp_shape = [None, self.image_size, self.image_size, 4]

        self.build_model(images_shape, labels_shape, labels_rgb_shape,
                         labels_rp_shape, labels_gp_shape, labels_bp_shape, config.is_train)

        counter = self.load(config.checkpoint_dir, restore=False)
        epoch_start = counter // batch_num
        batch_start = counter % batch_num

        global_step = tf.Variable(counter, trainable=False)
        learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, config.lr_decay_steps*batch_num, config.lr_decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        learning_step = optimizer.minimize(self.loss, global_step=global_step)

        tf.global_variables_initializer().run(session=self.sess)

        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter((os.path.join(config.checkpoint_dir, self.model_name, "log")), self.sess.graph)

        self.load(config.checkpoint_dir, restore=True)
        print("\nNow Start Training...\n")
        for ep in range(epoch_start, config.epoch):
            # Run by batch images
            for idx in range(batch_start, batch_num):

                batch_images, batch_labels, batch_labels_rgb, batch_labels_rp, batch_labels_gp, batch_labels_bp = \
                    get_batch(data_dir, data_num, config.batch_size)
                counter += 1

                #_, err, lr, = self.sess.run([learning_step, self.loss, learning_rate], feed_dict={
                #    self.images: batch_images, self.labels: batch_labels, self.rgb_labels: batch_labels_rgb,
                #    self.rp_labels: batch_labels_rp, self.gp_labels: batch_labels_gp, self.bp_labels:batch_labels_bp})
                _, err, lr, accuracy = self.sess.run([learning_step, self.loss, learning_rate, self.total_error], feed_dict={
                    self.images: batch_images, self.labels: batch_labels, self.rgb_labels: batch_labels_rgb,
                    self.rp_labels: batch_labels_rp, self.gp_labels: batch_labels_gp, self.bp_labels:batch_labels_bp})

                if counter % 10 == 0:
                    #print("Epoch: [%4d], batch: [%6d/%6d], loss: [%.8f], lr: [%.6f], step: [%d]" % ((ep+1), (idx+1), batch_num, err, lr, counter))
                    print("Epoch: [%4d], batch: [%6d/%6d], loss: [%.8f], accuracy: [%.8f], lr: [%.6f], step: [%d]" % ((ep+1), (idx+1), batch_num, err, accuracy, lr, counter))

                if counter % 1000 == 0:
                    self.save(config.checkpoint_dir, counter)

                    summary_str = self.sess.run(merged_summary_op, feed_dict={self.images: batch_images, self.labels: batch_labels, self.rgb_labels: batch_labels_rgb,self.rp_labels: batch_labels_rp, self.gp_labels: batch_labels_gp, self.bp_labels:batch_labels_bp})
                    summary_writer.add_summary(summary_str, counter)

                if counter > 0 and counter == batch_num * config.epoch:
                    self.save(config.checkpoint_dir, counter)
                    break
            self.save(config.checkpoint_dir, counter)
        self.save(config.checkpoint_dir, counter)
        summary_writer.close()

    def test(self, config):
        print("\nPrepare Data...\n")
        paths = prepare_data(config)
        data_num = len(paths)
        self.batch_size = 1
        avg_time = 0
        print("\nNow Start Testing...\n")
        for idx in range(data_num):
            data_name = paths[idx][4].split('\\')[3]
            
            gt_0 = cv2.imread(paths[idx][0])
            gt_45 = cv2.imread (paths[idx][1])
            gt_90 = cv2.imread (paths[idx][2])
            gt_135 = cv2.imread (paths[idx][3])
            input_ = cv2.imread(paths[idx][4], 0)
            input_ = np.reshape(input_, [input_.shape[0], input_.shape[1], 1])
            input_ = input_[np.newaxis, :]


            images_shape = input_.shape
            labels_shape = input_.shape * np.asarray([1, self.scale, self.scale, 12])
            labels_rgb_shape = input_.shape * np.asarray ([1, self.scale, self.scale, 3])
            labels_rp_shape = input_.shape * np.asarray ([1, self.scale, self.scale, 4])
            labels_gp_shape = input_.shape * np.asarray ([1, self.scale, self.scale, 4])
            labels_bp_shape = input_.shape * np.asarray ([1, self.scale, self.scale, 4])

            self.build_model(images_shape, labels_shape, labels_rgb_shape,
                         labels_rp_shape, labels_gp_shape, labels_bp_shape, config.is_train)
            tf.global_variables_initializer().run(session=self.sess) 

            self.load(config.checkpoint_dir, restore=True)

            time_ = time.time()
            result = self.sess.run([self.pred], feed_dict={self.images: input_ / 255.0})
            avg_time += time.time() - time_

            self.sess.close()
            tf.reset_default_graph()
            self.sess = tf.Session()

            x = np.squeeze(result) * 255.0
            x = np.clip(x, 0, 255)
            x = x[:, :, ::-1]
            x_0 = x[:, :, 0:3]
            x_45 = x[:, :, 3:6]
            x_90 = x[:, :, 6:9]
            x_135 = x[:, :, 9:12]

            #checkimage(np.uint8(x))

            save_dir = os.path.join (os.getcwd (), config.result_dir)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            imsave(x_0, save_dir + "/%s-Recon_0.png" % data_name)
            imsave (x_45, save_dir + "/%s-Recon_45.png" % data_name)
            imsave (x_90, save_dir + "/%s-Recon_90.png" % data_name)
            imsave (x_135, save_dir + "/%s-Recon_135.png" % data_name)
            imsave (gt_0, save_dir + "/%s-GT_0.png" % data_name)
            imsave (gt_45, save_dir + "/%s-GT_45.png" % data_name)
            imsave (gt_90, save_dir + "/%s-GT_90.png" % data_name)
            imsave (gt_135, save_dir + "/%s-GT_135.png" % data_name)

        print("Avg. Time:", avg_time / data_num)

    def load(self, checkpoint_dir, restore):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_path = str(ckpt.model_checkpoint_path)
            step = int(os.path.basename(ckpt_path).split('-')[1])
            if restore:
                self.saver.restore(self.sess, os.path.join(ckpt_path))
                print("\nCheckpoint Loading Success! %s\n" % ckpt_path)
        else:
            step = 0
            if restore:
                print("\nCheckpoint Loading Failed! \n")

        return step

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, "RDN.model"),
                        global_step=step)
