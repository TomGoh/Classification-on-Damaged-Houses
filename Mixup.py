import numpy as np
from matplotlib import pyplot as plt
from dataset_utils import load_test_vali_data, load_folder_data
import tensorflow as tf

BATCH_SIZE = 32


def generate_data():
    train_damage, place_holder1 = load_folder_data('archive/train_another/damage/', damaged=True)
    train_no_damage, place_holder2 = load_folder_data('archive/train_another/no_damage/', damaged=False)
    test_images, test_labels, test_abother_images, test_another_labels, vali_images, vali_labels = load_test_vali_data()
    train_images = np.array(train_no_damage + train_damage, dtype=np.float32) / 225.
    train_labels = np.array(place_holder2 + place_holder1, dtype=np.float32)
    train_images_1 = train_images.copy()
    train_labels_1 = train_labels.copy()
    state = np.random.get_state()
    np.random.shuffle(train_images_1)
    np.random.set_state(state)
    np.random.shuffle(train_labels_1)

    state = np.random.get_state()
    np.random.shuffle(vali_images)
    np.random.set_state(state)
    np.random.shuffle(vali_labels)

    train_ds_one = (
        tf.data.Dataset.from_tensor_slices((train_images, train_labels))
            .shuffle(BATCH_SIZE * 100)
            .batch(BATCH_SIZE)
    )
    train_ds_two = (
        tf.data.Dataset.from_tensor_slices((train_images_1, train_labels_1))
            .shuffle(BATCH_SIZE * 100)
            .batch(BATCH_SIZE)
    )

    train_ds = tf.data.Dataset.zip((train_ds_one, train_ds_two))
    val_ds = tf.data.Dataset.from_tensor_slices((np.array(vali_images,dtype=np.float32)/255., np.array(vali_labels))).batch(BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices((np.array(test_images,dtype=np.float32)/255., np.array(test_labels))).batch(BATCH_SIZE)
    test_another_ds = tf.data.Dataset.from_tensor_slices((np.array(test_abother_images,dtype=np.float32)/255., np.array(test_another_labels))).batch(BATCH_SIZE)

    train_ds_mu = train_ds.map(lambda ds_one, ds_two: mix_up(ds_one, ds_two, alpha=0.2),
                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return train_ds_mu, val_ds, test_ds, test_another_ds


def sample_beta_distribution(size, concentration_0, concentration_1):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    # Unpack two datasets
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    # Sample lambda and reshape it to do the mixup
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform mixup on both images and labels by combining a pair of images/labels
    # (one from each dataset) into one image/label
    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return images, labels


if __name__ == '__main__':
    train_ds_mu, val_ds, test_ds, test_another_ds = generate_data()
    sum_a=0
    sum_b=0
    for i, j in iter(train_ds_mu):
        a=np.sum(j.numpy()[:,0])
        sum_a=a+sum_a
        b=np.sum(j.numpy()[:,1])
        sum_b=sum_b+b
    print(sum_a,sum_b)
    sum_a = 0
    sum_b = 0
    for i, j in iter(val_ds):
        a = np.sum(j.numpy()[:, 0])
        sum_a = a + sum_a
        b = np.sum(j.numpy()[:, 1])
        sum_b = sum_b + b
    print(sum_a, sum_b)
