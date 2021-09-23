import numpy as np

np.set_printoptions(threshold=np.inf)
import tensorflow as tf
import cv2
from matplotlib import pyplot as plt

from dataset_utils import load_folder_data, load_test_vali_data
from ResNet import build_ResNet18


def gain_canny(image):
    image = image.astype(np.uint8)
    return cv2.Canny(image, 50, 200)


def gain_canny_data():
    BATCH_SIZE = 32
    train_damage, place_holder1 = load_folder_data('archive/train_another/damage/', damaged=True)
    train_no_damage, place_holder2 = load_folder_data('archive/train_another/no_damage/', damaged=False)
    train_damage_mask = list(map(lambda x: gain_canny(x), train_damage))
    train_no_damage_mask = list(map(lambda x: gain_canny(x), train_damage))
    train_mask = np.array(train_damage_mask + train_no_damage_mask, dtype=np.float32)
    test_images, test_labels, test_abother_images, test_another_labels, vali_images, vali_labels = load_test_vali_data()
    test_images_mask = list(map(lambda x: gain_canny(x), test_images))
    test_images_mask = np.array(test_images_mask, dtype=np.float32)
    test_abother_images_mask = list(map(lambda x: gain_canny(x), test_abother_images))
    test_abother_images_mask = np.array(test_abother_images_mask, dtype=np.float32)
    vali_images_mask = list(map(lambda x: gain_canny(x), vali_images))
    vali_images_mask = np.array(vali_images_mask, dtype=np.float32)
    train_images = np.array(train_no_damage + train_damage, dtype=np.float32)
    train_labels = np.array(place_holder2 + place_holder1, dtype=np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (gain_enhanced_data(train_images,train_mask),train_labels)).batch(BATCH_SIZE).shuffle(buffer_size=100)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (gain_enhanced_data(vali_images, vali_images_mask), vali_labels)).batch(
        BATCH_SIZE).shuffle(buffer_size=100)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (gain_enhanced_data(test_images, test_images_mask), test_labels)).batch(
        BATCH_SIZE).shuffle(buffer_size=100)
    test_another_ds = tf.data.Dataset.from_tensor_slices(
        (gain_enhanced_data(test_abother_images, test_abother_images_mask),
        test_another_labels)).batch(BATCH_SIZE).shuffle(buffer_size=100)

    return train_ds, val_ds, test_ds, test_another_ds


def gain_enhanced_data(dataset, mask):
    list = []
    for i in range(mask.shape[0]):
        list.append(np.expand_dims(mask[i], 2).repeat(3, axis=2))
    # print(np.array(list).shape)
    dataset1=dataset+np.array(list,dtype=np.float32)
    dataset1[dataset1>255]=255

    return dataset1/255.

if __name__ == '__main__':
    gain_canny_data()