import os
import numpy as np
from PIL import Image

def load_folder_data(dataset_path,damaged):
    filenames = os.listdir(dataset_path)
    images = []
    labels=[]
    if damaged:
        for i in filenames:
            img = np.array(Image.open((dataset_path + i)))
            images.append(img)
            labels.append([1,0])
    else:
        for i in filenames:
            img = np.array(Image.open((dataset_path + i)))
            images.append(img)
            labels.append([0,1])
    return images,labels

def load_test_vali_data():
    test_damage,test_damage_labels=load_folder_data('archive/test/damage/',True)
    test_no_damage,test_no_damage_labels=load_folder_data('archive/test/no_damage/',False)
    test_images=np.array(test_damage+test_no_damage,dtype=np.float32)
    test_labels=np.array(test_damage_labels+test_no_damage_labels,dtype=np.float32)

    test_another_damage,test_another_damage_labels = load_folder_data('archive/test_another/damage/', True)
    test_another_no_damage,test_another_no_damage_labels = load_folder_data('archive/test_another/no_damage/', False)
    test_abother_images = np.array(test_another_damage + test_another_no_damage,dtype=np.float32)
    test_another_labels=np.array(test_another_damage_labels+test_another_no_damage_labels,dtype=np.float32)

    vali_damage, vali_damage_labels = load_folder_data('archive/validation_another/damage/', True)
    vali_no_damage,vali_no_damage_labels = load_folder_data('archive/validation_another/no_damage/', False)
    vali_images = np.array(vali_damage + vali_no_damage,dtype=np.float32)
    vali_labels=np.array(vali_damage_labels+vali_no_damage_labels,dtype=np.float32)

    return test_images,test_labels,test_abother_images,test_another_labels,vali_images,vali_labels
