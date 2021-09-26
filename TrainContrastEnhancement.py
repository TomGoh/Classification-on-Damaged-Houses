import numpy as np
import tensorflow as tf

from ResNet import build_ResNet18
from SaveWeights import EarlyStoppingAtMinLoss


def modify_dataset(dataset_path,contrast):
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )

    train_data = train_gen.flow_from_directory(dataset_path + 'train_another'+'_'+contrast, batch_size=32,
                                               class_mode='categorical',
                                               shuffle=True, target_size=(128, 128))

    vali_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )

    vali_data = vali_gen.flow_from_directory(dataset_path + 'validation_another'+'_'+contrast, batch_size=32, class_mode='categorical',
                                             shuffle=True,
                                             target_size=(128, 128))

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )

    test_data = test_gen.flow_from_directory(dataset_path + 'test'+'_'+contrast, batch_size=32, class_mode='categorical',
                                             shuffle=True,
                                             target_size=(128, 128))

    another_test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
    )
    another_test_data = another_test_gen.flow_from_directory(dataset_path + 'test_another'+'_'+contrast, batch_size=32,
                                                             class_mode='categorical', shuffle=True,
                                                             target_size=(128, 128))
    return train_data, vali_data, test_data, another_test_data


if __name__ == '__main__':
    contrast_list=['ying','dhe','he']
    for i in contrast_list:
        train_data, vali_data, test_data, another_test_data = modify_dataset('archive/', contrast=i)
        resnet = build_ResNet18()
        EarlyStop = EarlyStoppingAtMinLoss(patience=10)
        resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        resnet_history = resnet.fit(train_data, validation_data=vali_data,
                                    epochs=50,
                                    callbacks=[EarlyStop], shuffle=True)
        test_loss, test_accuracy = resnet.evaluate(test_data)
        another_test_loss, another_test_accuracy = resnet.evaluate(another_test_data)
        np.savetxt('history/ValidationLoss_' + i + '.txt', resnet_history.history['val_loss'])
        np.savetxt('history/ValidationAccuracy_' + i + '.txt', resnet_history.history['val_accuracy'])
        np.savetxt('history/TrainingLoss_' + i + '.txt', resnet_history.history['loss'])
        np.savetxt('history/TrainingAccuracy_' + i + '.txt', resnet_history.history['accuracy'])
        np.savetxt('history/TestResult_' + i + '.txt',
                   np.array([test_loss, test_accuracy, another_test_loss, another_test_accuracy]))
