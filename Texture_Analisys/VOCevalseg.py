import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def eval_seg(exp_id, tset):
    '''
    Retorna acurácia de cada classe, acurácia média e matriz de confusão.

    exp_id: nome do experimento a ser avaliado
    tset: "train", "val" ou "test" set
    '''

    classes = [
               'aeroplane',
               'bicycle',
               'bird',
               'boat',
               'bottle',
               'bus',
               'car',
               'cat',
               'chair',
               'cow',
               'diningtable',
               'dog',
               'horse',
               'motorbike',
               'person',
               'pottedplant',
               'sheep',
               'sofa',
               'train',
               'tvmonitor']

    #  load  set
    images_ids = [line.rstrip('\n') for line in open(
        "./VOC2007/ImageSets/Segmentation/{}.txt".format(tset), "r")]

    # number of labels = number of classes plus one for the background
    num = 21
    confcounts = np.zeros((num * num))
    count = 0

    for i in range(len(images_ids)):
        imname = images_ids[i]

        # ground truth label file
        gtfile = "./VOC2007/SegmentationClass/{}.png".format(imname)
        gtim = np.array(Image.open(gtfile)).astype(np.float32)

        # results file
        resfile = "./results/VOC2007/Segmentation/{}_{}_cls/{}.png".format(exp_id, tset, imname)
        resim = np.array(Image.open(resfile)).astype(np.float32)

        # Check validity of results image
        maxlabel = np.max(resim)
        if maxlabel > 20:
            print("Results image ''{}'' has out of range value {} (the value should be <= 20".format(imname, maxlabel))
            return

        szgtim = gtim.shape
        szresim = resim.shape
        if szgtim!=szresim:
            print(
                "Results image ''{}'' is the wrong size, was {} x {}, should be {} x {}.".format(imname, szresim[0], szresim[1], szgtim[0], szgtim[1]))
            return
        
        # pixel locations to include in computation
        locs = gtim < 255

        # joint histogram
        sumim = 1+gtim+resim*num
        hs, b = np.histogram(sumim[locs], [x + 0.001 if x>0 else 0 for x in range(num*num + 1)])
        count += locs[locs != 0].size
        confcounts += hs

    # confusion matrix - first index is true label, second is inferred label
    conf = np.zeros((num,num))
    confcounts = confcounts.reshape((21, 21)).T
    rawcounts = confcounts
    overall_acc = 100*np.sum(np.diag(confcounts)) / np.sum(confcounts.flatten())
    print("Percentage of pixels correctly labelled overall: {}%".format(overall_acc))
    accuracies = np.zeros((21, 1))
    print('Percentage of pixels correctly labelled for each class')
    for j in range(num):
        rowsum = np.sum(confcounts[j]) 
        if rowsum > 0:
            conf[j] = 100*confcounts[j]/rowsum
        accuracies[j] = conf[j, j]
        clname = 'background'
        if j>0:
            clname = classes[j-1]
        print("  {}: {}%".format(clname, accuracies[j]))
    accuracies = accuracies[0:]
    avacc = np.mean(accuracies)
    print('-------------------------\n')
    print('Average accuracy: {}%'.format(avacc))

    return accuracies, avacc, conf


eval_seg("comp3", "val")