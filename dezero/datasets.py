import numpy as np
import math
import random
import os
import gzip

class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass

class DataLoader:
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_size = len(dataset)
        self.max_iter = math.ceil(self.data_size / batch_size)

        self.reset()

    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(len(self.dataset))
        else:
            self.index = np.arange(len(self.dataset))

    def __iter__(self):
        return self

    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration

        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i * batch_size: (i + 1) * batch_size] 
        batch = [self.dataset[i] for i in batch_index]
        x = np.array([example[0] for example in batch])
        t = np.array([example[1] for example in batch])

        self.iteration += 1
        return x, t

    def next(self):
        return self.__next__()


# MNIST
import urllib.request # 新增导入

def load_mnist(train=True, flatten=True):
    # ✅ AWS稳定镜像，pytorch官方使用
    url = 'https://ossci-datasets.s3.amazonaws.com/mnist/'
    key_file = {
        'train_img':'train-images-idx3-ubyte.gz',
        'train_label':'train-labels-idx1-ubyte.gz',
        'test_img':'t10k-images-idx3-ubyte.gz',
        'test_label':'t10k-labels-idx1-ubyte.gz'
    }
    dataset_dir = os.path.dirname(os.path.abspath(__file__)) + '/mnist'
    os.makedirs(dataset_dir, exist_ok=True)

    if train:
        img_file = key_file['train_img']
        label_file = key_file['train_label']
    else:
        img_file = key_file['test_img']
        label_file = key_file['test_label']
    
    # 文件不存在就下载
    for filename in [img_file, label_file]:
        file_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(file_path):
            print(f"正在下载 {filename} ...")
            urllib.request.urlretrieve(url + filename, file_path)

    # 读取图像
    with gzip.open(os.path.join(dataset_dir, img_file), 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    if flatten:
        images = images.reshape(-1, 784)
    else:
        images = images.reshape(-1, 1, 28, 28)
    # 读取标签
    with gzip.open(os.path.join(dataset_dir, label_file), 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return images, labels

class MNIST(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None):
        # ==========修复super传参，顺序匹配父类==========
        super().__init__(train, transform, target_transform)
        self.x, self.t = load_mnist(train=train, flatten=True)
        # 对接父类Dataset的self.data、self.label
        self.data = self.x
        self.label = self.t

    def __getitem__(self, idx):
        x, t = self.x[idx], self.t[idx]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            t = self.target_transform(t)
        return x, t

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    # 加载数据集，transform归一化像素 0~255 → 0~1
    train_set = MNIST(train=True, transform=lambda x: x / 255.0)
    test_set = MNIST(train=False, transform=lambda x: x / 255.0)
    print("训练集数量：", len(train_set))
    print("测试集数量：", len(test_set))

    # 取单样本
    x, t = train_set[0]
    print("单张图片shape:", x.shape)
    print("标签：", t)

    # DataLoader
    train_loader = DataLoader(train_set, batch_size=100, shuffle=True)
    for x_batch, t_batch in train_loader:
        print("batch x shape:", x_batch.shape)
        print("batch t shape:", t_batch.shape)
        break
