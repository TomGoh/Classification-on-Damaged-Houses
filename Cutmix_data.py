import random

import numpy as np
import os
from PIL import Image
from matplotlib import pyplot as plt


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
    test_images=np.array(test_damage+test_no_damage)
    test_labels=np.array(test_damage_labels+test_no_damage_labels)

    test_another_damage,test_another_damage_labels = load_folder_data('archive/test_another/damage/', True)
    test_another_no_damage,test_another_no_damage_labels = load_folder_data('archive/test_another/no_damage/', False)
    test_abother_images = np.array(test_another_damage + test_another_no_damage)
    test_another_labels=np.array(test_another_damage_labels+test_another_no_damage_labels)

    vali_damage, vali_damage_labels = load_folder_data('archive/validation_another/damage/', True)
    vali_no_damage,vali_no_damage_labels = load_folder_data('archive/validation_another/no_damage/', False)
    vali_images = np.array(vali_damage + vali_no_damage)
    vali_labels=np.array(vali_damage_labels+vali_no_damage_labels)

    return test_images,test_labels,test_abother_images,test_another_labels,vali_images,vali_labels


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(original_image, lam, mix_image, damaged):
    bbx1, bby1, bbx2, bby2 = rand_bbox(original_image.shape, lam)
    mask = np.ones_like(original_image)
    mask[bbx1:bbx2, bby1:bby2, :] = 0
    image = original_image * mask + mix_image * np.abs(1 - mask)
    ratio = np.sum(np.count_nonzero(mask)) / 128 / 128 / 3
    if damaged:
        probability = np.array([ratio, 1 - ratio])
    else:
        probability = np.array([1 - ratio, ratio])
    return image, probability


def apply_cutmix(damage_data, no_damage_data):
    post_images = []
    post_labels = []
    size = damage_data.shape[0]
    for i in damage_data:
        random_index = random.randint(0, size - 1)
        image, prob = cutmix(i, 0.7, no_damage_data[random_index], damaged=True)
        post_images.append(image)
        post_labels.append(prob)
    for i in no_damage_data:
        random_index = random.randint(0, size - 1)
        image, prob = cutmix(i, 0.7, damage_data[random_index], damaged=False)
        post_images.append(image)
        post_labels.append(prob)

    return np.array(post_images), np.array(post_labels)


def generate_train_data():
    train_damage,place_holder = load_folder_data('archive/train_another/damage/',damaged=True)
    train_no_damage,place_holder = load_folder_data('archive/train_another/no_damage/',damaged=False)
    post_image, post_label = apply_cutmix(np.array(train_damage), train_no_damage)
    return post_image, post_label

