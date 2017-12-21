# DEEP-LEARNING-3-AES-BIT
using deep learning to analysis 3-aes's first bit with 8 bits given

shuffle.txt中放了同一个密钥加密，前8比特遍历0到255，后124比特固定的三轮aes加密过程，aes遵从aes-128的加密规则，只截取了前3轮，将输出的128比特中的第一比特当做这8比特输入的标签

new_date_input.py中为读取每个训练及测试batch的代码


new_model.py中为模型主体，在全连接的基础上增加了LSTM的模块，并且用concat链接了相邻全连接层，最后为softmax层输出标签概率


train_and_test.py中封装了训练及测试相应的函数，训练使用了Adam优化器，学习率为1e^-3,batch为4，训练数据为248个，测试数据为8个。

train.py和test.py分别为训练与测试的代码，train中可以设置训练轮数，训练后会在本地生成checkpoint和tensorboard两个文件夹，分别对应保存的模型和训练中loss与acc的可视化数据


经过几轮迭代后，测试集上预测正确率可以达到0.6到0.8之间
