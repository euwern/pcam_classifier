# Breat Cancer Classifier on PatchCamelyon dataset
A Resnet classifier that trains on PatchCamelyon dataset 

# Dataset
PatchCamelyon dataset can be downloaded at https://github.com/basveeling/pcam
In this repo, we assume the hdf5 files are downloaded under "/mnt/datasets/pcam/", you should have the following hdf5 files:
- camelyonpatch_level_2_split_train_x.h5
- camelyonpatch_level_2_split_train_y.h5
- camelyonpatch_level_2_split_valid_x.h5
- camelyonpatch_level_2_split_valid_y.h5
- camelyonpatch_level_2_split_test_x.h5
- camelyonpatch_level_2_split_test_y.h5

# Training
```
#to train
python train.py
```
We obtained a Test Accuracy of 86.32%. 
