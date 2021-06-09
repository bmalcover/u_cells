#from __future__ import print_function

from typing import Tuple, Union
from enum import Enum
import glob
import os
import re

from matplotlib import pyplot as plt
import numpy as np 
import skimage.io as io
import skimage.transform as trans
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import utils as KU

CODES = [[128, 0, 0], [0, 128, 0], [0, 0, 128]]

class decode_mode(Enum):
    CELLS = 1
    CELLS_BCK = 2
    MULTICHANEL_N_BCK = 3
    MULTICHANEL_BCK = 4
    

def decode(mask, mode: decode_mode):
    """ Decodes the input image into a mask for class or a background image.
    """
    channels = []  
    if mode in (decode_mode.MULTICHANEL_N_BCK, decode_mode.MULTICHANEL_BCK):  
        for c in CODES:
            channel = np.argmax(c)
            mask_aux = mask[:, :, channel] == c[channel]
            mask_aux = mask_aux.astype(np.uint8) * 255

            channels.append(mask_aux)
        
        out_mask = np.dstack(channels)
    else:
        out_mask = (mask[:, :, 0] < 240) & (mask[:, :, 1] < 240) & (mask[:, :, 2] < 240)
        out_mask = out_mask.astype(np.uint8) * 255
        
        channels.append(out_mask)
        
    if mode in (decode_mode.MULTICHANEL_BCK, decode_mode.CELLS_BCK):            
        mask_bck = (mask[:, :, 0] > 240) & (mask[:, :, 1] > 240) & (mask[:, :, 2] > 240)
        mask_bck = mask_bck.astype(np.uint8) * 255
            
        channels.append(mask_bck)
        out_mask = np.dstack(channels)
        
    return out_mask


class DataGenerator(KU.Sequence):
    
    def __init__(self, batch_size:int, path:str, image_folder:str, mask_folder:str, aug_dict,
                 dcd_mode: Union[decode_mode, None], img_color_mode: str, mask_color_mode: str, target_size: Tuple[int, int], 
                 seed:int = 1):
        self.__batch_size = batch_size
        self.__path = path
        self.__decode_mode = dcd_mode
        
        image_datagen = ImageDataGenerator(**aug_dict)
        mask_datagen = ImageDataGenerator(**aug_dict)

        image_generator = image_datagen.flow_from_directory(
            path,
            classes=[image_folder],
            class_mode=None,
            color_mode=img_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            seed=seed
        )

        mask_generator = mask_datagen.flow_from_directory(
            path,
            classes=[mask_folder],
            class_mode=None,
            color_mode=mask_color_mode,
            target_size=target_size,
            batch_size=batch_size,
            seed=seed
        )
    
        self.__image_generator = image_generator
        self.__mask_generator = mask_generator
        self.__generator = self.__get_merged_info()
        
    def __get_merged_info(self):
        for img, mask in zip(self.__image_generator, self.__mask_generator):
            img = img / 255

            if self.__decode_mode is not None:
                if self.__decode_mode is decode_mode.CELLS:
                    new_mask = np.zeros((mask.shape[0],mask.shape[1], mask.shape[2]))

                    for m_idx in range(0, mask.shape[0]):
                        m = mask[m_idx, :, :]
                        new_mask[m_idx, :, :] = decode(m, self.__decode_mode)
                else:
                    new_mask = np.zeros((mask.shape[0],mask.shape[1], mask.shape[2], dcd_mode.value))

                    for m_idx in range(0, mask.shape[0]):
                        m = mask[m_idx, :, :, :]
                        new_mask[m_idx, :, :, :] = decode(m, self.__decode_mode)
                mask = new_mask

            mask = mask / 255
            yield img, mask            
    
    def __len__(self):
        return len((self.__image_generator))

    def __getitem__(self, idx):
        return next(self.__generator)

    
    
def encode(channels: np.ndarray):
    img_res = np.zeros((channels.shape[0], channels.shape[1], 3), dtype=np.uint8)

    d = channels.shape[-1]

    for i in range(0, d):
        channel = channels[:, :, i]
        img_res[channel == 255] = CODES[i]

    return img_res


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
