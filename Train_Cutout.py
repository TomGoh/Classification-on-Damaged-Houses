import numpy as np
import tensorflow as tf

from Cutout import cutout_preprocessing
from ResNet import build_ResNet18
from SaveWeights import EarlyStoppingAtMinLoss


def modify_generator(preprocessing_function,dataset_path):
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=preprocessing_function,
    )

    train_data = train_gen.flow_from_directory(dataset_path + 'train_another', batch_size=32, class_mode='categorical',
                                               shuffle=True, target_size=(128, 128))

    vali_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=None,
    )

    vali_data = vali_gen.flow_from_directory(dataset_path + 'validation_another', batch_size=32, class_mode='categorical', shuffle=True,
                                             target_size=(128, 128))

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=None,
    )

    test_data = test_gen.flow_from_directory(dataset_path + 'test', batch_size=32, class_mode='categorical', shuffle=True,
                                             target_size=(128, 128))

    another_test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=None,
    )
    another_test_data = another_test_gen.flow_from_directory(dataset_path + 'test_another', batch_size=32,
                                                             class_mode='categorical', shuffle=True,
                                                             target_size=(128, 128))
    return train_data, vali_data, test_data, another_test_data


if __name__ == '__main__':
    train_data, vali_data, test_data, another_test_data = modify_generator(preprocessing_function=cutout_preprocessing,dataset_path = 'archive/')
    ResNet_Model = build_ResNet18()

    ResNet_Model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                         loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    ResNet_Model.summary()
    # EarlyStop = tf.keras.callbacks.EarlyStopping(patience=10)
    resnet_history = ResNet_Model.fit_generator(train_data, validation_data=vali_data, epochs=50, shuffle=True,
                                                callbacks=[EarlyStoppingAtMinLoss(patience=10)])
    test_loss,test_accuracy = ResNet_Model.evaluate(test_data)
    another_test_loss,another_test_accuracy = ResNet_Model.evaluate(another_test_data)
    np.savetxt('history/ValidationLoss_Cutout.txt', resnet_history.history['val_loss'])
    np.savetxt('history/ValidationAccuracy_Cutout.txt', resnet_history.history['val_accuracy'])
    np.savetxt('history/TrainingLoss_Cutout.txt', resnet_history.history['loss'])
    np.savetxt('history/TrainingAccuracy_Cutout.txt', resnet_history.history['accuracy'])
    np.savetxt('history/TestResult_Cutout.txt', np.array([test_loss,test_accuracy,another_test_loss,another_test_accuracy]))