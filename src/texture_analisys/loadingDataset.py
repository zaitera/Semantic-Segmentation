from utils import *


def imgs_from_category(dataset):
    filename = os.path.join(set_dir, dataset + ".txt")
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        header=None,
        names=['filename'])
    return df

def imgs_from_category_as_list(dataset):
    df = imgs_from_category(dataset)
    return df['filename'].values

def annotation_file_from_img(img_name):
	global ann_dir
	return os.path.join(ann_dir, str(img_name)) + '.xml'

def load_annotation(img_filename):
    xml = ""
    with open(annotation_file_from_img(img_filename)) as f:
        xml = f.readlines()
    xml = ''.join([line.strip('\t') for line in xml])
    return BeautifulSoup(xml, features="html5lib")

def get_all_obj_and_box(objname, img_set):
    img_list = imgs_from_category_as_list(objname, img_set)
    
    for img in img_list:
        annotation = load_annotation(img)

def load_img(img_filename):
    return io.load_image(os.path.join(img_dir, img_filename + '.jpg'))

def load_train_data(trainTest):

	if not os.path.exists(root_dir+'csvs'):
	    os.mkdir(root_dir+'csvs')
	    print("Directory " , root_dir+'csvs' ,  " Created ")

		
	train_filename = root_dir + 'csvs/' + trainTest + '.csv'
	if os.path.isfile(train_filename):
		return pd.read_csv(train_filename)
	else:
		train_img_list = imgs_from_category_as_list(trainTest)
		formatter = "{:06d}".format
		train_img_list = list(map(formatter, train_img_list))
		data = []
		for item in train_img_list:
			anno = load_annotation(item)
			objs = anno.findAll('object')
			for obj in objs:
				obj_names = obj.findChildren('name')
				for name_tag in obj_names:
					fname = anno.findChild('filename').contents[0]
					bbox = obj.findChildren('bndbox')[0]
					xmin = int(bbox.findChildren('xmin')[0].contents[0])
					ymin = int(bbox.findChildren('ymin')[0].contents[0])
					xmax = int(bbox.findChildren('xmax')[0].contents[0])
					ymax = int(bbox.findChildren('ymax')[0].contents[0])
					name = str(name_tag).replace('<name>','')
					name = name.replace('</name>', '')
					labObj = labels.index(name)
					data.append([fname, xmin, ymin, xmax, ymax, labObj])
		df = pd.DataFrame(data, columns=['fname', 'xmin', 'ymin', 'xmax', 'ymax', 'obj'])
		df.to_csv(train_filename, index=False)
		return df
