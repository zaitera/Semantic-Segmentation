# Running DeepLab on PASCAL VOC 2012 Semantic Segmentation Dataset

This page walks through the steps required to run DeepLab on PASCAL VOC 2012 on
a local machine.

## Download dataset and convert to TFRecord

We have prepared the script (under the folder `datasets`) to download and
convert PASCAL VOC 2012 semantic segmentation dataset to TFRecord.

```bash
# From the tensorflow/models/research/deeplab/datasets directory.
sh download_and_convert_voc2012.sh
```

The converted dataset will be saved at
./deeplab/datasets/pascal_voc_seg/tfrecord

## Recommended Directory Structure for Training and Evaluation

```
+ datasets
  + pascal_voc_seg
    + VOCdevkit
      + VOC2012
        + JPEGImages
        + SegmentationClass
    + tfrecord
    + exp
      + train_on_train_set
        + train
        + eval
        + vis
```

## It is highly recommanded to use the .yaml conda environment provided in these repository to avoid compatibility problems, check conda websites to see how to import it
## Check the scripts in the deeplab/scripts folder, for each type of test theres an easily understandable script.
