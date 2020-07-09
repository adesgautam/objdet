
from PIL import Image
import os
import pandas as pd
from Utils.convert_format import convert_vott_csv_to_yolo

# Set paths
input_images = "training_images"
model_folder =  "models"
output_csv =  input_images+'/annotations-export.csv'
output_yolo_format = input_images+'/data_train.txt'
classes_filename = 'Utils/data_classes.txt'

#Prepare the dataset for YOLO
multi_df = pd.read_csv(output_csv)
print(multi_df.shape)
labels = multi_df['label'].unique()
print("labels:", labels)
labeldict = dict(zip(labels,range(len(labels))))
print("labeldict:", labeldict)
multi_df.drop_duplicates(subset=None, keep='first', inplace=True)
print(multi_df.shape)
convert_vott_csv_to_yolo(multi_df, labeldict, path = input_images, target_name=output_yolo_format)

# Make classes file
file = open(classes_filename, "w") 

#Sort Dict by Values
SortedLabelDict = sorted(labeldict.items() ,  key=lambda x: x[1])
for elem in SortedLabelDict:
    file.write(elem[0]+'\n') 
file.close() 

print("Conversion completed!")