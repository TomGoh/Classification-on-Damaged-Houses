import tensorflow as tf
import numpy as np

from ResNet import build_ResNet18
from SaveWeights import EarlyStoppingAtMinLoss
from Mixup import generate_data

if __name__ == '__main__':
    train_ds_mu, val_ds, test_ds, test_another_ds = generate_data()
    resnet = build_ResNet18()
    EarlyStop = EarlyStoppingAtMinLoss(patience=10)
    resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
    resnet_history = resnet.fit(train_ds_mu, validation_data=val_ds,
                                epochs=50,
                                callbacks=[EarlyStop], shuffle=True)
    test_loss, test_accuracy = resnet.evaluate(test_ds)
    another_test_loss, another_test_accuracy = resnet.evaluate(test_another_ds)
    np.savetxt('history/ValidationLoss_Mixup.txt', resnet_history.history['val_loss'])
    np.savetxt('history/ValidationAccuracy_Mixup.txt', resnet_history.history['val_accuracy'])
    np.savetxt('history/TrainingLoss_Mixup.txt', resnet_history.history['loss'])
    np.savetxt('history/TrainingAccuracy_Mixup.txt', resnet_history.history['accuracy'])
    np.savetxt('history/TestResult_Mixup.txt', np.array([test_loss, test_accuracy, another_test_loss, another_test_accuracy]))
