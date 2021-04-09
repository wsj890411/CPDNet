import cv2
import numpy as np
import h5py
import tensorflow as tf
import math
import glob
from random import shuffle
import os 

def tf_keepGradient_loss(img1, img2):
    """Re implement the exclusion loss in perceptual loss, CVPR2018 """

    def compute_gradient(img):
        gradx = img[:, 1:, :, :] - img[:, :-1, :, :]
        grady = img[:, :, 1:, :] - img[:, :, :-1, :]
        return gradx, grady

    gradx1, grady1 = compute_gradient(img1)
    gradx2, grady2 = compute_gradient(img2)
    x_mse = tf.losses.absolute_difference(gradx1, gradx2)
    y_mse = tf.losses.absolute_difference(grady1, grady2)
    loss_gradxy = x_mse+y_mse

    return loss_gradxy
    
def rgb2ycbcr(img):
    y = 16 + (65.481 * img[:, :, 0]) + (128.553 * img[:, :, 1]) + (24.966 * img[:, :, 2])
    return y / 255

def PSNR(target, ref, scale):
    target_data = np.array(target, dtype=np.float32)
    ref_data = np.array(ref, dtype=np.float32)

    target_y = rgb2ycbcr(target_data)
    ref_y = rgb2ycbcr(ref_data)
    diff = ref_y - target_y

    shave = scale
    diff = diff[shave:-shave, shave:-shave]

    mse = np.mean((diff / 255) ** 2)
    if mse == 0:
        return 100

    return -10 * math.log10(mse)

# def imread(path):
#     img = cv2.imread(path)
#     return img

def imsave(image, path):
    cv2.imwrite(os.path.join(os.getcwd(),path),image)

def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)

def modcrop(img, scale =3):
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int((h / scale)) * scale
        w = int((w / scale)) * scale
        img = img[0:h, 0:w]
    return img


def preprocess(path, scale = 3, eng = None, mdouble = None):

    img_0 = cv2.imread(path[0])
    img_45 = cv2.imread(path[1])
    img_90 = cv2.imread(path[2])
    img_135 = cv2.imread(path[3])
    mosaic = cv2.imread(path[4], 0)
    h, w = mosaic.shape

    gt = np.concatenate((img_0, img_45, img_90, img_135), axis = 2)
    input = mosaic
    #input = np.concatenate((mosaic, mosaic, mosaic, mosaic), axis=2)

    input_ = modcrop(input, scale)
    label_ = modcrop(gt, scale)

    if eng is None:
        input_ = cv2.resize(input_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC)

    else:
        input_ = np.asarray(eng.imresize(mdouble(label_.tolist()), 1.0/scale, 'bicubic'))

    input_ = np.reshape(input_,(h,w,1))
    label_ = label_[:, :, ::-1]

    return input_, label_

def make_data_hf(input_, label_, config, times):
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'train.h5')
    else:
        savepath = os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'test.h5')

    if times == 0:
        if os.path.exists(savepath):
            print("\n%s have existed!\n" % (savepath))
            return False
        else:
            hf = h5py.File(savepath, 'w')

            if config.is_train:
                input_h5 = hf.create_dataset("input", (1, config.image_size, config.image_size, 1),
                                            maxshape=(None, config.image_size, config.image_size, 1),
                                            chunks=(1, config.image_size, config.image_size, 1), dtype='float32')
                label_h5 = hf.create_dataset("label", (1, config.image_size*config.scale, config.image_size*config.scale, config.c_dim), 
                                            maxshape=(None, config.image_size*config.scale, config.image_size*config.scale, config.c_dim), 
                                            chunks=(1, config.image_size*config.scale, config.image_size*config.scale, config.c_dim),dtype='float32')
            else:
                input_h5 = hf.create_dataset("input", (1, input_.shape[0], input_.shape[1], input_.shape[2]), 
                                            maxshape=(None, input_.shape[0], input_.shape[1], input_.shape[2]), 
                                            chunks=(1, input_.shape[0], input_.shape[1], input_.shape[2]), dtype='float32')
                label_h5 = hf.create_dataset("label", (1, label_.shape[0], label_.shape[1], label_.shape[2]), 
                                            maxshape=(None, label_.shape[0], label_.shape[1], label_.shape[2]), 
                                            chunks=(1, label_.shape[0], label_.shape[1], label_.shape[2]),dtype='float32')
    else:
        hf = h5py.File(savepath, 'a')
        input_h5 = hf["input"]
        label_h5 = hf["label"]

    if config.is_train:
        input_h5.resize([times + 1, config.image_size, config.image_size, 1])
        input_h5[times : times+1] = input_
        label_h5.resize([times + 1, config.image_size*config.scale, config.image_size*config.scale, config.c_dim])
        label_h5[times : times+1] = label_
    else:
        input_h5.resize([times + 1, input_.shape[0], input_.shape[1], input_.shape[2]])
        input_h5[times : times+1] = input_
        label_h5.resize([times + 1, label_.shape[0], label_.shape[1], label_.shape[2]])
        label_h5[times : times+1] = label_

    hf.close()
    return True

def make_sub_data(data, config):
    if config.matlab_bicubic:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        mdouble = matlab.double
    else:
        eng = None
        mdouble = None

    times = 0
    for i in range(len(data)):
        input_, label_, = preprocess(data[i], config.scale, eng, mdouble)
        if len(input_.shape) == 3:
            h, w, c = input_.shape
        else:
            h, w = input_.shape

        for x in range(0, h * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
            for y in range(0, w * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                sub_label = label_[x: x + config.image_size * config.scale, y: y + config.image_size * config.scale]
                
                sub_label = sub_label.reshape([config.image_size * config.scale , config.image_size * config.scale, config.c_dim])

                # r_gxy = 0
                # for d in range(0, 4):
                #     t = cv2.cvtColor(sub_label[:, :, d*3:d*3+3], cv2.COLOR_BGR2YCR_CB)
                #     t = t[:, :, 0]
                #     gx = t[1:, 0:-1] - t[0:-1, 0:-1]
                #     gy = t[0:-1, 1:] - t[0:-1, 0:-1]
                #     Gxy = (gx**2 + gy**2)**0.5
                #     r_gxy = r_gxy + float((Gxy > 10).sum()) / ((config.image_size*3)**2) * 100
                # if r_gxy/4 < 10:
                #     continue

                sub_label =  sub_label / 255.0

                x_i = int(x / config.scale)
                y_i = int(y / config.scale)
                sub_input = input_[x_i: x_i + config.image_size, y_i: y_i + config.image_size]
                #sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                sub_input = sub_input / 255.0

                # checkimage(sub_input)
                # checkimage(sub_label)

                save_flag = make_data_hf(sub_input, sub_label, config, times)
                if not save_flag:
                    return
                times += 1

        print("image: [%2d], total: [%2d], patches: [%2d]"%(i, len(data), times))

    if config.matlab_bicubic:
        eng.quit()


def prepare_data(config):
    if config.is_train:
        data = []
        data_dir = os.path.join(os.path.join(os.path.pardir, config.train_set_dir), config.train_set)
        dataset_dir_list = os.listdir (data_dir)
        for data_name in dataset_dir_list:
            gt_dir = os.path.join(data_dir, data_name)
            for gt_name in os.listdir (gt_dir):
                if gt_name.split ('_')[1] == str (0):
                    gt_0_dir = os.path.join(data_dir, data_name, gt_name, data_name) + '.png'
                elif gt_name.split ('_')[1] == str (45):
                    gt_45_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
                elif gt_name.split ('_')[1] == str (90):
                    gt_90_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
                elif gt_name.split ('_')[1] == str (135):
                    gt_135_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
                else:
                    gt_mosaic_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
            data.append([gt_0_dir, gt_45_dir, gt_90_dir, gt_135_dir, gt_mosaic_dir])
    else:
        if config.test_img != "":
            data = [os.path.join(os.getcwd(), config.test_img)]
        else:
            # data_dir = os.path.join(os.path.join(os.getcwd(), "Test"), config.test_set)
            # data = glob.glob(os.path.join(data_dir, "*.bmp"))
            data = []
            data_dir = os.path.join (os.path.join (os.path.pardir, config.test_set_dir), config.test_set)
            dataset_dir_list = os.listdir (data_dir)
            for data_name in dataset_dir_list:
                gt_dir = os.path.join (data_dir, data_name)
                for gt_name in os.listdir (gt_dir):
                    if gt_name.split ('_')[1] == str (0):
                        gt_0_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
                    elif gt_name.split ('_')[1] == str (45):
                        gt_45_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
                    elif gt_name.split ('_')[1] == str (90):
                        gt_90_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
                    elif gt_name.split ('_')[1] == str (135):
                        gt_135_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
                    else:
                        gt_mosaic_dir = os.path.join (data_dir, data_name, gt_name, data_name) + '.png'
                data.append ([gt_0_dir, gt_45_dir, gt_90_dir, gt_135_dir, gt_mosaic_dir])
    return data

def input_setup(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """

    data = prepare_data(config)
    shuffle(data)
    make_sub_data(data, config)
    return data

def augmentation(batch, random):
    if random[0] < 0.3:
        batch_flip = np.flip(batch, 1)
    elif random[0] > 0.7:
        batch_flip = np.flip(batch, 2)
    else:
        batch_flip = batch

    if random[1] < 0.5:
        batch_rot = np.rot90(batch_flip, 1, [1, 2])
    else:
        batch_rot = batch_flip

    return batch_rot

def get_data_dir(checkpoint_dir, is_train):
    if is_train:
        return os.path.join(os.path.join(os.getcwd(), checkpoint_dir), 'train.h5')
    else:
        return os.path.join(os.path.join(os.getcwd(), checkpoint_dir), 'test.h5')

def get_data_num(path):
     with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        return input_.shape[0]


def get_batch(path, data_num, batch_size):
    with h5py.File(path, 'r') as hf:
        input_ = hf['input']
        label_ = hf['label']

        random_batch = np.random.rand(batch_size) * (data_num - 1)
        batch_images = np.zeros([batch_size, input_[0].shape[0], input_[0].shape[1], input_[0].shape[2]])
        batch_labels = np.zeros([batch_size, label_[0].shape[0], label_[0].shape[1], label_[0].shape[2]])
        batch_labels_rgb = np.zeros ([batch_size, label_[0].shape[0], label_[0].shape[1], 3])
        batch_labels_rp = np.zeros ([batch_size, label_[0].shape[0], label_[0].shape[1], 4])
        batch_labels_gp = np.zeros ([batch_size, label_[0].shape[0], label_[0].shape[1], 4])
        batch_labels_bp = np.zeros ([batch_size, label_[0].shape[0], label_[0].shape[1], 4])

        for i in range(batch_size):

            batch_images[i, :, :, :] = np.asarray(input_[int(random_batch[i])])
            batch_labels[i, :, :, :] = np.asarray(label_[int(random_batch[i])])
            tmp = label_[int(random_batch[i])]
            rgb_image = get_rgb(tmp, label_[0].shape[0], label_[0].shape[1])
            g_image = np.stack([tmp[:, :, 0], tmp[:, :, 3], tmp[:, :, 6], tmp[:, :, 9]], 2)
            b_image = np.stack([tmp[:, :, 1], tmp[:, :, 4], tmp[:, :, 7], tmp[:, :, 10]], 2)
            r_image = np.stack([tmp[:, :, 2], tmp[:, :, 5], tmp[:, :, 8], tmp[:, :, 11]], 2)
            batch_labels_rgb[i, :, :, :] = np.asarray(rgb_image)
            batch_labels_rp[i, :, :, :] = np.asarray (r_image)
            batch_labels_gp[i, :, :, :] = np.asarray (g_image)
            batch_labels_bp[i, :, :, :] = np.asarray (b_image)

        random_aug = np.random.rand(2)
        batch_images = augmentation(batch_images, random_aug)
        batch_labels = augmentation(batch_labels, random_aug)
        batch_labels_rgb = augmentation (batch_labels_rgb, random_aug)
        batch_labels_rp = augmentation (batch_labels_rp, random_aug)
        batch_labels_gp = augmentation (batch_labels_gp, random_aug)
        batch_labels_bp = augmentation (batch_labels_bp, random_aug)
        return batch_images, batch_labels, batch_labels_rgb, batch_labels_rp, batch_labels_gp, batch_labels_bp


def get_rgb(raw, w, h):
    mask_0 = np.zeros([w, h])
    mask_45 = np.zeros([w, h])
    mask_90 = np.zeros([w, h])
    mask_135 = np.zeros([w, h])

    for i in range(0, w):
        for j in range(0, h):
            if i % 2 == 0 and j % 2 == 0:
                mask_0[i, j] = 1
            elif i % 2 == 0 and j % 2 == 1:
                mask_45[i, j] = 1
            elif i % 2 == 1 and j % 2 == 0:
                mask_135[i, j] = 1
            elif i % 2 == 1 and j % 2 == 1:
                mask_90[i, j] = 1

    img_g = raw[:, :, 0] * mask_0 + raw[:, :, 3] * mask_45 + raw[:, :, 6] * mask_90 + raw[:, :, 9] * mask_135
    img_b = raw[:, :, 1] * mask_0 + raw[:, :, 4] * mask_45 + raw[:, :, 7] * mask_90 + raw[:, :, 10] * mask_135
    img_r = raw[:, :, 2] * mask_0 + raw[:, :, 5] * mask_45 + raw[:, :, 8] * mask_90 + raw[:, :, 11] * mask_135
    rgb = np.stack([img_r, img_b, img_r], 2)

    return rgb


def get_image(path, scale, matlab_bicubic):
    if matlab_bicubic:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        mdouble = matlab.double
    else:
        eng = None
        mdouble = None
    
    image, label = preprocess(path, scale, eng, mdouble)
    image = image[np.newaxis, :]
    label = label[np.newaxis, :]

    if matlab_bicubic:
        eng.quit()

    return image, label
