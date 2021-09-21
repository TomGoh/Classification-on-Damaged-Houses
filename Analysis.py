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

    plt.subplot(211)
    plt.plot(cutmix_training_loss,linewidth=5,label='Cutmix Training Loss')
    plt.plot(cutmix_vali_loss,linewidth=5,label='Cutmix Valid Loss')
    plt.plot(cutout_training_loss, linewidth=5,label='Cutout Training Loss')
    plt.plot(cutout_vali_loss, linewidth=5,label='Cutout Valid Loss')
    plt.title('Loss of Cutmix and Cutout')
    plt.legend()

    plt.subplot(212)
    plt.plot(cutmix_training_acc, linewidth=5,label='Cutmix Training Accuracy')
    plt.plot(cutmix_vali_acc, linewidth=5,label='Cutmix Valid Accuracy')
    plt.plot(cutout_training_acc, linewidth=5,label='Cutout Training Accuracy')
    plt.plot(cutout_vali_acc, linewidth=5,label='Cutout Valid Accuracy')
    plt.title('Accuracy of Cutmix and Cutout')
    plt.legend()

    plt.tight_layout()
    plt.show()

    X = np.arange(2)

    test_acc=np.array([9.039999842643737793e-01,8.809999823570251465e-01])
    test_another_ac=np.array([9.124444723129272461e-01,9.039999842643737793e-01])
    x_ticks = ['Cutout','Cutmix']
    plt.bar(X - 0.125, test_acc - 0.88, width=0.25, label='Test')
    plt.bar(X + 0.125, test_another_ac - 0.88, width=0.25, label='Test Another')
    for x, y in enumerate(test_acc):
        plt.text(x - 0.125, y - 0.88, '%s' % round(y, 4), ha='center')
    for x, y in enumerate(test_another_ac):
        plt.text(x + 0.125, y - 0.88, '%s' % round(y, 4), ha='center')
    # plt.bar(X+0.1,test_data_loss,width=0.2)
    # plt.bar(X+0.3,test_another_data_loss,width=0.2)
    plt.xticks(X, x_ticks)
    plt.yticks(np.arange(0.9, 1, 0.1))
    plt.title('Accuaccy over test and test_another')
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0), borderaxespad=0.)

    plt.show()

