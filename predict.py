import os
import pickle
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import random

# from model import ConvNet
from ResNet_101 import ResNet101
from hwdb import HWDB


def predict(net, image, key_list) -> str:
    outputs = net.forward(image, 'cpu')
    # 取得分最高的那个类
    _, predicted = torch.max(outputs.data, 1)
    return key_list[predicted]


if __name__ == "__main__":
    # 读取分类类别
    with open('char_dict', 'rb') as f:
        class_dict = pickle.load(f)
    list_of_keys = list(class_dict.keys())
    num_classes = len(class_dict)

    # 加载模型与参数
    net = ResNet101(num_classes)
    # if torch.cuda.is_available():
    #     net = net.cuda()
    net.load_state_dict(torch.load('./checkpoints/ResNet101_019_3060ti.pth'))

    # 输入并预测
    image_folder_dir = "./data/predict/"
    image_dir_list = os.listdir(image_folder_dir)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        # transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    for image_dir in image_dir_list:
        image = Image.open(image_folder_dir + image_dir)
        image = transform(image)
        if image.shape[0] == 3:
            image = image.resize(1, 3, 64, 64)
        else:
            image = image[:-1].resize(1, 3, 64, 64)
        plt.imshow(image[0][0], cmap='gray')
        plt.show()
        print(predict(net, image, list_of_keys))

    # 数据集随机取样测试
    # transform = transforms.Compose([
    #     transforms.Resize((64, 64)),
    #     transforms.ToTensor(),
    # ])
    # dataset = HWDB(path=r'data', transform=transform)
    # index = random.randint(0, dataset.train_size)
    # img = dataset.get_sample(index)[0]
    # plt.imshow(img[0], cmap='gray')
    # plt.show()
    # outputs = net(img.resize(1, 3, 64, 64))
    # # 取得分最高的那个类
    # _, predicted = torch.max(outputs.data, 1)
    # print(list_of_keys[predicted])

