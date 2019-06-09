"""
================
RAG Thresholding
================

This example constructs a Region Adjacency Graph (RAG) and merges regions
which are similar in color. We construct a RAG and define edges as the
difference in mean color. We then join regions with similar mean color.
"""

from skimage import data, io, segmentation, color
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2

img = cv2.imread('./testFiles/000122.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

labels1 = segmentation.slic(img,  compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')

g = graph.rag_mean_color(img, labels1)
labels2 = graph.cut_threshold(labels1, g, 29)
out2 = color.label2rgb(labels2, img, kind='avg')

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True,
                       figsize=(6, 8))

ax[0].imshow(out1)
ax[1].imshow(out2)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()

color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
