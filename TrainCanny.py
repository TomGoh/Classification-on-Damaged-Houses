import tensorflow as tf
import numpy as np

from ResNet import build_ResNet18
from SaveWeights import EarlyStoppingAtMinLoss
from Canny import generate_data

if __name__ == '__main__':
    train_images,train_labels,vali_images,vali_labels,test_images,test_labels,test_another_images,test_antoher_labels = generate_data()
    resnet = build_ResNet18()
    EarlyStop = EarlyStoppingAtMinLoss(patience=10)
    resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    resnet_history = resnet.fit(train_images,train_labels, validation_data=(vali_images,vali_labels),
                                epochs=50,
                                batch_size=32,
                                callbacks=[EarlyStop], shuffle=True)
    test_loss, test_accuracy = resnet.evaluate(test_images,test_labels)
    another_test_loss, another_test_accuracy = resnet.evaluate(test_another_images,test_antoher_labels)
    np.savetxt('history/ValidationLoss_Canny.txt', resnet_history.history['val_loss'])
    np.savetxt('history/ValidationAccuracy_Canny.txt', resnet_history.history['val_accuracy'])
    np.savetxt('history/TrainingLoss_Canny.txt', resnet_history.history['loss'])
    np.savetxt('history/TrainingAccuracy_Canny.txt', resnet_history.history['accuracy'])
    np.savetxt('history/TestResult_Canny.txt', np.array([test_loss, test_accuracy, another_test_loss, another_test_accuracy]))
