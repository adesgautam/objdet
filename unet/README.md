# U-NET: CNN for Instance Segmentation
## Implementation of UNET model in Keras.

## Original paper [here](https://arxiv.org/pdf/1505.04597.pdf)

It requires very less data to train on your custom data. Min 30 images would suffice.

The demo data is in *dataset* directory. It is from [ISBI challenge](http://brainiac2.mit.edu/isbi_challenge/). The preprocessing and loading of data using ImageDataGenerator using Keras, is in *data_utils.py*.

The model architecture is in *unet_model.py*. It has image input size of (256,256). You can increase it in *train.ipynb*, but it should be a multiple of 32.


