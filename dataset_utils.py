import os
import numpy as np
from PIL import Image
import tensorflow as tf

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

def build_data_generator(dataset_path,para_list, preprocessing_function=None, flip=True):
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=para_list.get('rotation_range'),
        rescale=1. / 255,
        width_shift_range=para_list.get('width_shift_range'),
        height_shift_range=para_list.get('height_shift_range'),
        zoom_range=para_list.get('zoom_range'),
        horizontal_flip=flip,
        vertical_flip=flip,
        preprocessing_function=preprocessing_function,
    )

    train_data = train_gen.flow_from_directory(dataset_path + 'train_another', batch_size=32, class_mode='categorical',
                                               shuffle=True, target_size=(128, 128))

    vali_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=para_list.get('rotation_range'),
        rescale=1. / 255,
        width_shift_range=para_list.get('width_shift_range'),
        height_shift_range=para_list.get('height_shift_range'),
        zoom_range=para_list.get('zoom_range'),
        horizontal_flip=flip,
        vertical_flip=flip,
        preprocessing_function=preprocessing_function,
    )

    vali_data = vali_gen.flow_from_directory(dataset_path + 'validation_another', batch_size=32,
                                             class_mode='categorical', shuffle=True,
                                             target_size=(128, 128))

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=para_list.get('rotation_range'),
        rescale=1. / 255,
        width_shift_range=para_list.get('width_shift_range'),
        height_shift_range=para_list.get('height_shift_range'),
        zoom_range=para_list.get('zoom_range'),
        horizontal_flip=flip,
        vertical_flip=flip,
        preprocessing_function=preprocessing_function,
    )

    test_data = test_gen.flow_from_directory(dataset_path + 'test', batch_size=32, class_mode='categorical',
                                             shuffle=True,
                                             target_size=(128, 128))

    another_test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=para_list.get('rotation_range'),
        rescale=1. / 255,
        width_shift_range=para_list.get('width_shift_range'),
        height_shift_range=para_list.get('height_shift_range'),
        zoom_range=para_list.get('zoom_range'),
        horizontal_flip=flip,
        vertical_flip=flip,
        preprocessing_function=preprocessing_function,
    )
    another_test_data = another_test_gen.flow_from_directory(dataset_path + 'test_another', batch_size=32,
                                                             class_mode='categorical', shuffle=True,
                                                             target_size=(128, 128))
    return train_data, vali_data, test_data, another_test_data
