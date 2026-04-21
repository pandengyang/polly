import os
import numpy as np
from PIL import Image


class Dataset:

    def __init__(self, path_dataset):
        self.path_dataset = path_dataset

    def toMatrix(self):
        # 获取所有bmp文件，按序号排序
        files = [
            f for f in os.listdir(self.path_dataset) if f.endswith('.bmp')
        ]

        def file_key(f):
            return int(f.split('-')[0])

        files.sort(key=file_key)

        datas = []
        actual_results = []
        for fname in files:
            label = int(fname.split('-')[1].split('.')[0])
            img_path = os.path.join(self.path_dataset, fname)
            img = Image.open(img_path).convert('L')
            arr = np.array(img)

            pixels = []
            for y in range(4):
                for x in range(3):
                    v = arr[y, x]
                    pixels.append(1 if v < 128 else 0)
            datas.append(pixels)

            if label == 0:
                actual_results.append([1, 0])

            else:
                actual_results.append([0, 1])

        datas = np.array(datas)
        actual_results = np.array(actual_results)

        return datas, actual_results
