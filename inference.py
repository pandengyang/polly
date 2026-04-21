import numpy as np
import os
from PIL import Image
from activation_function.sigmoid import Sigmoid

if __name__ == '__main__':
    # 1. 加载模型参数
    params = np.load('polly.npz')
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # 2. 加载测试数据
    dataset_dir = './dataset_test'
    files = [f for f in os.listdir(dataset_dir) if f.endswith('.bmp')]
    files.sort(key=lambda f: int(f.split('-')[0]))

    X = []
    filenames = []
    for fname in files:
        img_path = os.path.join(dataset_dir, fname)
        img = Image.open(img_path).convert('L')
        arr = np.array(img)
        pixels = []
        for y in range(4):
            for x in range(3):
                v = arr[y, x]
                pixels.append(1 if v < 128 else 0)
        X.append(pixels)
        filenames.append(fname)
    X = np.array(X)

    # 3. 初始化激活函数
    actfunc = Sigmoid()

    # 4. 推理
    Z1 = X @ W1 + b1
    A1 = actfunc.function(Z1)
    Z2 = A1 @ W2 + b2
    A2 = actfunc.function(Z2)

    # 5. 打印推理结果
    for fname, prob in zip(filenames, A2):
        print(f"{fname}\t是 0 的概率: {prob[0]:.4f}\t是 1 的概率: {prob[1]:.4f}")