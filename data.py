#from __future__ import print_function

from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
from matplotlib import pyplot as plt
import re


CODES = [[128, 0, 0], [0, 128, 0], [0, 0, 128]]


def decode(mask):
    
    channels = []  # back.astype(np.uint8) * 255]
    for c in CODES:
        channel = np.argmax(c)
        aux = mask[:, :, channel] == c[channel]
        aux = aux.astype(np.uint8) * 255

        channels.append(aux)

    #back = ((mask[:, :, 0] == 255) & (mask[:, :, 1] == 255) & (mask[:, :, 2] == 255))
    #back = back.astype(np.uint8) * 255

    #channels.append(back)
    
    return np.dstack(channels)


def encode(channels: np.ndarray):
    img_res = np.zeros((channels.shape[0], channels.shape[1], 3), dtype=np.uint8)

    d = channels.shape[-1]

    for i in range(0, d):
        channel = channels[:, :, i]
        img_res[channel == 255] = CODES[i]

    return img_res


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict,
                   image_color_mode="grayscale", mask_color_mode="grayscale",
                   image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class = False, num_class = 2, save_to_dir=None,
                   target_size=(256, 256), seed=1, decode_flag=False):
    '''
    Generation of an image and a mask at the same time
    Use the same seed for image_datagen and mask_datagen to ensure the
    transformation for image and mask is the same
    To visualize the results of generator, set save_to_dir = "your path"
    '''

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)

    for img, mask in train_generator:
        img = img / 255

        if decode_flag:
            new_mask = np.zeros((1,mask.shape[1], mask.shape[2], 3))
            for m_idx in range(0, 1):
                m = mask[m_idx, :, :, :]
                new_mask[m_idx, :, :, :] = decode(m)
            mask = new_mask

        mask = mask / 255
        yield img, mask


def testGenerator(test_path, target_size=(256, 256), flag_multi_class=False,
                  as_gray=True, image=True):

    filenames = glob.glob(test_path)
    filenames.sort(key=lambda f: int(re.sub('\D', '', f)))
    # filenames = sorted(filenames)
    for filename in filenames:

        img = io.imread(filename,as_gray = as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,)+img.shape)

        yield img


def visualize_label(num_class, color_dict, img):

    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    return img_out / 255


def save_result(save_path, npyfile, size=None, flag_encode=False):

    for i, item in enumerate(npyfile):
        if flag_encode:
            img = encode(item)
        else:
            img = item
        # img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,:,0]
        if size is not None:
            img = cv2.resize(img, size)
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)
