import numpy as np
import cv2

from dataset_utils import load_folder_data, load_test_vali_data


def canny(image):
    image = image.astype(np.uint8)
    return cv2.Canny(image, 50, 200)



def enhance_data(dataset, mask):
    mask[mask==255]=100
    list = []
    for i in range(mask.shape[0]):
        list.append(np.expand_dims(mask[i], 2).repeat(3, axis=2))
    # print(np.array(list).shape)
    dataset1=dataset+np.array(list,dtype=np.float32)
    dataset1[dataset1>255]=255

    return dataset1/255.


def generate_data():
    BATCH_SIZE=32
    train_damage, place_holder1 = load_folder_data('archive/train_another/damage/', damaged=True) # list
    train_no_damage,place_holder2=load_folder_data('archive/train_another/no_damage/',damaged=False) #list
    train_images=np.array(train_damage+train_no_damage,dtype=np.float32)
    train_labels=np.array(place_holder1+place_holder2,dtype=np.float32)

    test_images, test_labels, test_another_images, test_another_labels, vali_images, vali_labels = load_test_vali_data()

    train_mask=np.array(list(map(lambda x: canny(x), train_images)),dtype=np.float32)
    vali_mask=np.array(list(map(lambda  x: canny(x),vali_images)),dtype=np.float32)
    test_mask = np.array(list(map(lambda x: canny(x), test_images)), dtype=np.float32)
    test_anther_mask = np.array(list(map(lambda x: canny(x), test_another_images)), dtype=np.float32)

    return enhance_data(train_images,train_mask),train_labels,enhance_data(vali_images,vali_mask),\
           vali_labels,enhance_data(test_images,test_mask),test_labels,enhance_data(test_another_images,test_anther_mask),\
           test_another_labels

