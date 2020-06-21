import os
import glob
import numpy as np 
import skimage.io as io
import skimage.transform as trans

from keras.preprocessing.image import ImageDataGenerator

red = [255,0,0]
green = [0,255,0]
blue = [0,0,255]
yellow = [255,255,0]
magenta = [255,0,255]
cyan = [0,255,255]

COLOURS_DICT = [red, green, blue, yellow, magenta, cyan]

def adjustData(img, mask, flag_multi_class, num_class):
    img = img / 255
    if(flag_multi_class):
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=False, num_class=1, save_to_dir=None, target_size=(256,256), seed=1):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img,mask)

def process_image(test_path, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    img = io.imread(os.path.join(test_path,"%d.png"%i), as_gray = as_gray)
    img = img / 255
    img = trans.resize(img, target_size)
    img = np.reshape(img, (img.shape[0], img.shape[1], 1)) if (not flag_multi_class) else img
    img = np.reshape(img, (1, img.shape[0], img.shape[2], img.shape[2]))
    return img

def postprocess_img(prediction, flag_multi_class=False, num_classes=1):
    if flag_multi_class:
        if len(prediction.shape) == 3:
            img = prediction[:,:,0]
        else:
            img = prediction

        img_out = np.zeros((img.shape[0], img.shape[1], 3))
        for i in range(num_classes):
            img_out[img == i,:] = COLOURS_DICT[i]
        img =  img_out / 255
    else:   
        img = prediction[:,:,0]

