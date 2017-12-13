from train_and_test import train1,test1
import os

if not os.path.exists('tensorboard/'):
    os.makedirs('tensorboard/')
if not os.path.exists('checkpoint/'):
    os.makedirs('checkpoint/')
train1(122)

