# Simplified Matterport's version of mask RCNN

###The original implementation of Matterport is (here)[https://github.com/matterport/Mask_RCNN]

This is just a simplified version of Matterport's version with some little bells and whistles.

Additions:
* A Flask based UI for easy testing of the trained model.
* Conversion of dataset from CSV to JSON exported from the VIA tool(used for annotation). This is useful if you want to distribute the annotation for multiple people and then compile all the annotations into JSON and split it for train and val.
