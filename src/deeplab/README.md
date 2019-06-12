# DeepLab: Deep Labelling for Semantic Image Segmentation

DeepLab is a state-of-art deep learning model for semantic image segmentation,
where the goal is to assign semantic labels (e.g., person, dog, cat and so on)
to every pixel in the input image. Current implementation includes the following
features:


*  DeepLabv3+ [1]: Extended DeepLabv3 to include a simple yet effective
    decoder module to refine the segmentation results especially along object
    boundaries. Furthermore, in this encoder-decoder structure one can
    arbitrarily control the resolution of extracted encoder features by atrous
    convolution to trade-off precision and runtime.

### Potentially based on https://github.com/tensorflow/models/tree/master/research/deeplab
### Use the conda env environment.yaml provided for a better compatibility 
#### For more instrucitons (database related and commands for running) check:  
           ./g3doc/"the readme related for your case"
* This repository and results were tested on the PASCAL 2007 database.  

* "xception65_coco_voc_trainaug" area the pretrained parameter used in our case

#### For citations:
*   DeepLabv3+:

```
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```

*  Architecture search for dense prediction cell:

```
@inproceedings{dpc2018,
  title={Searching for Efficient Multi-Scale Architectures for Dense Image Prediction},
  author={Liang-Chieh Chen and Maxwell D. Collins and Yukun Zhu and George Papandreou and Barret Zoph and Florian Schroff and Hartwig Adam and Jonathon Shlens},
  booktitle={NIPS},
  year={2018}
}

```

*  Auto-DeepLab (also called hnasnet in core/nas_network.py):

```
@inproceedings{autodeeplab2019,
  title={Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic
Image Segmentation},
  author={Chenxi Liu and Liang-Chieh Chen and Florian Schroff and Hartwig Adam
  and Wei Hua and Alan Yuille and Li Fei-Fei},
  booktitle={CVPR},
  year={2019}
}

```
* Current model implementation

  ###  Xception [9, 10]: A powerful network structure intended for server-side deployment.


This directory contains TensorFlow [11] implementation. codes
allowe users to train the model, evaluate results in terms of mIOU (mean
intersection-over-union), and visualize segmentation results. PASCAL VOC
2012 [12] and Cityscapes [13] are used; 
#### But for this repository, an adaptation has been done to run the codes on PASCAL VOC 2007 too.

```
1.  **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](https://arxiv.org/abs/1802.02611). In ECCV, 2018.
```
