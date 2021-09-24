import numpy as np
import os

from dataset_utils import load_folder_data,load_test_vali_data

def raw_data():
    train_damage, place_holder1 = load_folder_data('archive/train_another/damage/', damaged=True)  # list
    train_no_damage, place_holder2 = load_folder_data('archive/train_another/no_damage/', damaged=False)  # list
    train_images = np.array(train_damage + train_no_damage, dtype=np.float32)/255.
    train_labels = np.array(place_holder1 + place_holder2, dtype=np.float32)

    test_images, test_labels, test_another_images, test_another_labels, vali_images, vali_labels = load_test_vali_data()

    return train_images,train_labels,vali_images/255.,vali_labels,test_images/255.,test_labels,test_another_images/255.,test_another_labels