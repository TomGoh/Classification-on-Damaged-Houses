import numpy as np
import tensorflow as tf

from ResNet import build_ResNet18
from dataset_utils import build_data_generator
from SaveWeights import EarlyStoppingAtMinLoss

if __name__ == '__main__':

    para_dir1 = {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 'zoom_range': 0}
    para_dir2 = {'rotation_range': 54, 'width_shift_range': 0, 'height_shift_range': 0, 'zoom_range': 0}
    para_dir3 = {'rotation_range': 0, 'width_shift_range': 0.15, 'height_shift_range': 0.15, 'zoom_range': 0}
    para_dir4 = {'rotation_range': 0, 'width_shift_range': 0, 'height_shift_range': 0, 'zoom_range': 0.15}

    para_dir = list([para_dir1, para_dir2, para_dir3, para_dir4])
    for i, j in zip(para_dir, range(1, 5)):
        resnet = build_ResNet18()
        EarlyStop = EarlyStoppingAtMinLoss(patience=10)
        resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        train_data, vali_data, test_data, another_test_data = build_data_generator(dataset_path='archive/',
                                                                                   para_list=i,
                                                                                   preprocessing_function=None,
                                                                                   flip=False)
        resnet_history = resnet.fit(train_data, validation_data=vali_data, epochs=50, shuffle=True,
                                    callbacks=[EarlyStoppingAtMinLoss(patience=10)])
        test_loss, test_accuracy = resnet.evaluate(test_data)
        another_test_loss, another_test_accuracy = resnet.evaluate(another_test_data)

        np.savetxt('history/ValidationLoss_Generator' + str(j) + '.txt', resnet_history.history['val_loss'])
        np.savetxt('history/ValidationAccuracy_Generator' + str(j) + '.txt', resnet_history.history['val_accuracy'])
        np.savetxt('history/TrainingLoss_Generator' + str(j) + '.txt', resnet_history.history['loss'])
        np.savetxt('history/TrainingAccuracy_Generator' + str(j) + '.txt', resnet_history.history['accuracy'])
        np.savetxt('history/TestResult_Generator' + str(j) + '.txt',
                   np.array([test_loss, test_accuracy, another_test_loss, another_test_accuracy]))
