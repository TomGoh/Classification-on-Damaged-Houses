import tensorflow as tf
import numpy as np

from ResNet import build_ResNet18
from SaveWeights import EarlyStoppingAtMinLoss
from Cutmix_data import generate_train_data, load_test_vali_data


def generate_data():
    train_images, train_labels = generate_train_data()
    test_images, test_labels, test_abother_images, test_another_labels, vali_images, vali_labels = load_test_vali_data()
    return train_images/255., train_labels, test_images/255., test_labels, test_abother_images/255., test_another_labels, vali_images/255., vali_labels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels, test_abother_images, test_another_labels, vali_images, vali_labels = generate_data()
    resnet = build_ResNet18()
    resnet.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=1e-4),
                   loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    resnet_history = resnet.fit(train_images, train_labels, batch_size=32,validation_data=(vali_images, vali_labels), epochs=50,
                                callbacks=[EarlyStoppingAtMinLoss(patience=10)])
    test_loss,test_accuracy = resnet.evaluate(test_images,test_labels)
    another_test_loss,another_test_accuracy = resnet.evaluate(test_abother_images,test_another_labels)
    np.savetxt('ValidationLoss_Cutmix.txt', resnet_history.history['val_loss'])
    np.savetxt('ValidationAccuracy_Cutmix.txt', resnet_history.history['val_accuracy'])
    np.savetxt('TrainingLoss_Cutmix.txt', resnet_history.history['loss'])
    np.savetxt('TrainingAccuracy_Cutmix.txt', resnet_history.history['accuracy'])
    np.savetxt('TestResult_Cutmix.txt', np.array([test_loss,test_accuracy,another_test_loss,another_test_accuracy]))