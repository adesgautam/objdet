
import os
import numpy as np
import pickle
import keras.backend as K

from PIL import Image
from time import time
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from Utils.keras_yolo3.yolo3.model import preprocess_true_boxes, yolo_body, tiny_yolo_body, yolo_loss
from Utils.keras_yolo3.yolo3.utils import get_random_data
from Utils.train_utils import get_classes, get_anchors, create_model, create_tiny_model, data_generator, data_generator_wrapper, ChangeToOtherMachine

# Setup directories and filenames
YOLO_train_file = 'training_images/data_train.txt'
classes_file = 'Utils/data_classes.txt'
log_dir = 'models/logs'
anchors_path = 'Utils/yolo_anchors.txt'
weights_path = 'models/yolo.h5'


## Hyperparameters
# multiple of 32, height, width
input_shape = (416, 416)
val_split = 0.1
is_tiny = False
random_seed = 0
batch_size = 32
epochs = 30
lr1 = 1e-3
lr2 = 1e-4
epoch1, epoch2 = epochs, epochs 

## Checkpoint settings
save_best_model_only = True
model_save_per_epochs = 1

## Fine tuning settings
fine_tune = True
ft_batch_size = 4
ft_epochs = epoch1+epoch2

# Set random seed
np.random.seed(random_seed)

class_names = get_classes(classes_file)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

# is_tiny_version = (len(anchors)==6) # default setting
if is_tiny:
    model = create_tiny_model(input_shape, anchors, num_classes, freeze_body=2, weights_path = weights_path)
else:
    model = create_model(input_shape, anchors, num_classes, freeze_body=2, weights_path = weights_path) # make sure you know what you freeze

# Setup logging
log_dir_time = os.path.join(log_dir,'{}'.format(int(time())))
logging = TensorBoard(log_dir = log_dir_time)

# Create callbacks
checkpoint = ModelCheckpoint(os.path.join(log_dir,'checkpoint.h5'),
                                            monitor = 'val_loss', 
                                            save_weights_only = True, 
                                            save_best_only = save_best_model_only, 
                                            period = model_save_per_epochs)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

# Split train and validation data
val_split = val_split
with open(YOLO_train_file) as f:
    lines = f.readlines()

# This step makes sure that the path names correspond to the local machine
# This is important if annotation and training are done on different machines (e.g. training on AWS)
lines  = ChangeToOtherMachine(lines, remote_machine = '')
np.random.shuffle(lines)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val

# Train with frozen layers first, to get a stable loss.
# Adjust num epochs to your dataset. This step is enough to obtain a decent model.

## Train Stage 1 (for making a stable model first)
model.compile(optimizer=Adam(lr=lr1), loss={
    # use custom yolo_loss Lambda layer.
    'yolo_loss': lambda y_true, y_pred: y_pred})

print('Training on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train//batch_size),
                            validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                            validation_steps=max(1, num_val//batch_size),
                            epochs=epochs,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])

# Save stage 1 model by with loss in filename
step1_train_loss = history.history['loss']
step1_val_loss = history.history['val_loss']

last_train_loss = step1_train_loss[-2] if step1_train_loss[-1]=='' else step1_train_loss[-1]
last_val_loss = step1_val_loss[-2] if step1_val_loss[-1]=='' else step1_val_loss[-1]

# Save stage 1 model
model.save_weights(os.path.join(log_dir,'trained_weights_stage_1_trainloss_'+ 
                                        str(last_train_loss) + '_valloss_' + str(last_val_loss) + '.h5'))

# Log training loss
file = open(os.path.join(log_dir_time,'step1_loss.npy'), "w")
with open(os.path.join(log_dir_time,'step1_loss.npy'), 'w') as f:
    for item in step1_train_loss:
        f.write("%s\n" % item) 
file.close()

# Log validation loss
file = open(os.path.join(log_dir_time,'step1_val_loss.npy'), "w")
with open(os.path.join(log_dir_time,'step1_val_loss.npy'), 'w') as f:
    for item in step1_val_loss:
        f.write("%s\n" % item) 
file.close()


## Fine tuned the stage 1 model    
# Unfreeze and continue training, to fine-tune.
# Train longer if the result is unsatisfactory.
if fine_tune:
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    model.compile(optimizer=Adam(lr=lr2), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
    print('Unfreezing all layers...')

    batch_size = ft_batch_size # note that more GPU memory is required after unfreezing the body
    print('Training on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    history = model.fit_generator(data_generator_wrapper(lines[:num_train], batch_size, input_shape, anchors, num_classes),
                                steps_per_epoch=max(1, num_train//batch_size),
                                validation_data=data_generator_wrapper(lines[num_train:], batch_size, input_shape, anchors, num_classes),
                                validation_steps=max(1, num_val//batch_size),
                                epochs=ft_epochs,
                                initial_epoch=epoch1,
                                callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    # Save fine tuned model by with loss in filename
    step2_train_loss = history.history['loss']
    step2_val_loss = history.history['val_loss']

    last_train_loss = step2_train_loss[-2] if step2_train_loss[-1]=='' else step2_train_loss[-1]
    last_val_loss = step2_val_loss[-2] if step2_val_loss[-1]=='' else step2_val_loss[-1]

    # Save fine tuned model
    model.save_weights(os.path.join(log_dir,'trained_weights_final_finetuned_'+ 
                                            str(last_train_loss) + '_valloss_' + str(last_val_loss) + '.h5'))
    
    # Log training loss
    file = open(os.path.join(log_dir_time,'step2_loss.npy'), "w")
    with open(os.path.join(log_dir_time,'step2_loss.npy'), 'w') as f:
        for item in step2_train_loss:
            f.write("%s\n" % item) 
    file.close()
    
    file = open(os.path.join(log_dir_time,'step2_val_loss.npy'), "w")
    with open(os.path.join(log_dir_time,'step2_val_loss.npy'), 'w') as f:
        for item in step2_val_loss:
            f.write("%s\n" % item) 
    file.close()
