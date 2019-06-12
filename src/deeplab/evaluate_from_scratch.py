
"""Results generator script for the DeepLab model.
    @author: Abdullah Zaiter 
    @date: 11/06/2019
"""
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from PIL import Image



import tensorflow as tf

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('checkpoint_path', None, 'Directory of model checkpoints.')
flags.DEFINE_string('set_type', None, 'Directory of model checkpoints.')
flags.DEFINE_string('exp_name', None, 'Directory of model checkpoints.')




class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)
    
  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]

LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])


FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
MODEL = DeepLabModel(FLAGS.checkpoint_path)


def run_visualization(dir_name):
  """Inferences DeepLab model and visualizes result."""
  import os
  os.chdir(dir_name)
  images_ids = [line.rstrip('\n') for line in open(
        "VOC2007/ImageSets/Segmentation/{}.txt".format(FLAGS.set_type), "r")]
  os.chdir("./VOC2007/JPEGImages/")
  imagesN = [i+".jpg" for i in images_ids]
  images = list(map(Image.open, imagesN))
  results = list(map(MODEL.run, images))
  results = [i.astype(np.uint8) for i in results]

  imgs_seg = list(map(Image.fromarray, results))
  try:
    os.mkdir("../Segmentation/{}_{}_cls".format(FLAGS.exp_name,FLAGS.set_type))
    print("Directory Created ") 
  except FileExistsError:
    print("Directory already exists")
  os.chdir("../Segmentation/{}_{}_cls".format(FLAGS.exp_name,FLAGS.set_type))
  [imgs_seg[i].save("{}.png".format(images_ids[i])) for i in range(len(imgs_seg))]





if __name__ == '__main__':
    flags.mark_flag_as_required('checkpoint_path')
    flags.mark_flag_as_required('set_type')
    flags.mark_flag_as_required('exp_name')
    run_visualization("./deeplab/datasets/pascal_voc_seg/VOCdevkit")