# minst数据集的载入

from torch.utils.data import Dataset
import gzip
import os
import numpy as np


# #加载训练集数据
# data_train = MNIST('./data', download=True, transform=transforms.Compose([transforms.Resize((23, 32)), transforms.ToTensor()]))
#
# #加载测试集数据
# data_test = MNIST('./data', train=False, download=True, transform=transforms.Compose([transforms.Resize((23, 32)), transforms.ToTensor()]))


# 设置训练集和测试机载入器
# data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
# data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

# 自定义数据集类型
def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder, label_name), 'rb') as lbpath:  # rb表示的是读取二进制数据
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28)
    return x_train, y_train


class MyDataset(Dataset):
    def __init__(self, folder, data_name, label_name, transform=None):
        # (data_set, labels) = torch.load(folder, data_name, label_name)
        (data_set, labels) = load_data(folder, data_name, label_name)
        self.data_set = data_set
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data_set[index], int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.data_set)

# #测试
# # 实例化数据集类型
# trainDataset = MyDataset('data/MNIST/raw', 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', transform=transforms.ToTensor())
#
# # 加载数据集
# train_loader = DataLoader(dataset=trainDataset, batch_size=10, shuffle=False)
#
# #显示单张图片
# images, labels = next(iter(train_loader))
# img = torchvision.utils.make_grid(images)
#
# img = img.numpy().transpose(1, 2, 0)
# std = [0.5, 0.5, 0.5]
# mean = [0.5, 0.5, 0.5]
# img = img*std + mean
# print(labels)
# plt.imshow(img)
# plt.show()
