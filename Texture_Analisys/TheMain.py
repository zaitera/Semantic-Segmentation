from teste2 import *
from loadingDataset import *

def main():

    print("Getting the textures descriptors")
    df = load_train_data('train')        
    for row_num, entry in df.iterrows():
        img = cv2.imread(img_dir + '/' +entry.fname)
        res, bg = filtering_image(img)
        props = comatImg(res)
        if (row_num==0):
            trainProps = props
            trainCat = entry.obj
        else:
            trainProps = np.vstack((trainProps, props))
            trainCat = np.vstack((trainCat, entry.obj))
        if (row_num < 130):
            prop_bg = comatImg(bg)            
            trainProps = np.vstack((trainProps, prop_bg))
            trainCat = np.vstack((trainCat, 0))
    print("Done!")
        
    trainProps = np.array(trainProps)
    trainCat = np.array(trainCat)   
    #name = input("Select a image on the test database: ")
    #img = cv2.imread(img_dir_test + '/' + name)
    #GT = cv2.imread(gt_dir_test + '/' + name.replace('.jpg', '.png'))
    #segments = segmentations(img)
    print("-------------")
    print("Time for testing")
    clf_svm, clf_rf = begTest(trainProps, trainCat)
    df = load_train_data('val')

    for row_num, entry in df.iterrows():
        img = cv2.imread(img_dir + '/' +entry.fname)
        test(img, entry.fname, clf_rf, clf_svm)
    print("Done! All the .txt have been created!")
    
 

if __name__ == '__main__':

    main()