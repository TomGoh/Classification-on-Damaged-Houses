import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    cutmix_training_acc=np.loadtxt('TrainingAccuracy_Cutmix.txt')
    cutmix_training_loss=np.loadtxt('TrainingLoss_Cutmix.txt')
    cutmix_vali_acc=np.loadtxt('ValidationAccuracy_Cutmix.txt')
    cutmix_vali_loss=np.loadtxt('ValidationLoss_Cutmix.txt')

    cutout_training_acc = np.loadtxt('TrainingAccuracy_Cutout.txt')
    cutout_training_loss = np.loadtxt('TrainingLoss_Cutout.txt')
    cutout_vali_acc = np.loadtxt('ValidationAccuracy_Cutout.txt')
    cutout_vali_loss = np.loadtxt('ValidationLoss_Cutout.txt')

    mixup_training_acc=np.loadtxt('TrainingAccuracy_Mixup.txt')
    mixup_training_loss=np.loadtxt('TrainingLoss_Mixup.txt')
    mixup_vali_acc=np.loadtxt('ValidationAccuracy_Mixup.txt')
    mixup_vali_loss=np.loadtxt('ValidationLoss_Mixup.txt')


    plt.subplot(211)
    plt.plot(cutmix_training_loss,linewidth=5,label='Cutmix Training Loss')
    plt.plot(cutmix_vali_loss,linewidth=5,label='Cutmix Valid Loss')
    plt.plot(cutout_training_loss, linewidth=5,label='Cutout Training Loss')
    plt.plot(cutout_vali_loss, linewidth=5,label='Cutout Valid Loss')
    plt.plot(mixup_training_loss, linewidth=5, label='Mixup Training Loss')
    plt.plot(mixup_vali_loss, linewidth=5, label='Mixup Valid Loss')
    plt.title('Loss of Cutmix and Cutout')
    plt.legend()

    plt.subplot(212)
    plt.plot(cutmix_training_acc, linewidth=5,label='Cutmix Training Accuracy')
    plt.plot(cutmix_vali_acc, linewidth=5,label='Cutmix Valid Accuracy')
    plt.plot(cutout_training_acc, linewidth=5,label='Cutout Training Accuracy')
    plt.plot(cutout_vali_acc, linewidth=5,label='Cutout Valid Accuracy')
    plt.plot(mixup_training_acc, linewidth=5, label='Mixup Training Accuracy')
    plt.plot(mixup_vali_acc, linewidth=5, label='Mixup Valid Accuracy')

    plt.title('Accuracy of Cutmix and Cutout')
    plt.legend()

    plt.tight_layout()
    plt.show()

    X = np.arange(3)

    cutout_test_data=np.loadtxt('TestResult_Cutout.txt')
    cutmix_test_data=np.loadtxt('TestResult_Cutmix.txt')
    mixup_test_data=np.loadtxt('TestResult_Mixup.txt')
    test_acc=np.array([cutout_test_data[1],cutmix_test_data[1],mixup_test_data[1]])
    test_another_acc=np.array([cutout_test_data[-1],cutmix_test_data[-1],mixup_test_data[-1]])
    x_ticks = ['Cutout','Cutmix','Mixup']
    plt.bar(X - 0.125, test_acc - 0.93, width=0.25, label='Test')
    plt.bar(X + 0.125, test_another_acc - 0.93, width=0.25, label='Test Another')
    for x, y in enumerate(test_acc):
        plt.text(x - 0.125, y - 0.93, '%s' % round(y, 4), ha='center')
    for x, y in enumerate(test_another_acc):
        plt.text(x + 0.125, y - 0.93, '%s' % round(y, 4), ha='center')
    # plt.bar(X+0.1,test_data_loss,width=0.2)
    # plt.bar(X+0.3,test_another_data_loss,width=0.2)
    plt.xticks(X, x_ticks)
    plt.yticks(np.arange(0.9, 1, 0.1))
    plt.title('Accuaccy over test and test_another')
    plt.legend()

    plt.show()

