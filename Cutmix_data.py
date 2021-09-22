import random

import numpy as np
from dataset_utils import load_folder_data



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

