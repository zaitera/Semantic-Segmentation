from teste2 import *
from loadingDataset import *

def main():


    all_files = os.listdir(set_dir)
    image_sets = sorted(list(set([filename.replace('.txt', '').strip().split('_')[0] for filename in all_files])))
    print ("\n {} \n".format(image_sets))
    
    for cat in image_sets[:5]:
        print("Calculating features for >", cat)  
        df = load_train_data(cat)
        i = 0
        for row_num, entry in df.iterrows():
            img = cv2.imread(img_dir + '/' +entry.fname)
            res = filtering_image(img)
            props = comatImg(res)            
            if (i==0):
                trainProps = props
                trainCat = cat
            else:
                trainProps = np.vstack((trainProps, props))
                trainCat = np.vstack((trainCat, cat))
            i+=1

    name = input("Select a image on the test database: ")
    img = cv2.imread(img_dir_test + '/' + name)
    segments = segmentations(img)
    
    
    test(trainProps, trainCat, segments)
    



    






    
    

if __name__ == '__main__':
    main()