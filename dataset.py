from torch.utils.data import Dataset
from torchvision.io import read_image
import os

# 从一个文件夹中创建数据集，文件格式{name}.{label}.{suffix} （使用.分隔文件名，类型和扩展名）


class cls_dataset(Dataset):

    type_num = 6

    def __init__(self, root, transform) -> None:
        super().__init__()
        self.root = root
        self.dict = {
            'q': 0,  # 晴
            'e': 1,  # 阴
            'r': 2,  # 雨
            'a': 3,  # 雪
            's': 4,  # 雾
            'f': 5   # 夜晚
        }
        self.set = []

        for root, dirs, files in os.walk(self.root):
            target = root.split('\\')[-1]
            print(root, target)
            for file in files:
                type_ = file.split('.')[-2]
                t = self.dict[type_]
                if (type_ != 'd'):
                    pic = read_image(os.path.join(root, file))
                    pic = transform(pic)
                    information = {
                        'image': pic,
                        'target': t
                    }
                    self.set.append(information)

    def __getitem__(self, index):
        img, target = self.set[index]['image'], self.set[index]['target']
        return img, target

    def __len__(self):
        return len(self.set)
    def __str__(self):
        count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for item in self.set:
            target = item['target']
            count[target] += 1
        return f"图片数量：晴({count[0]}) 阴({count[1]}) 雨({count[2]}) 雪({count[3]}) 雾({count[4]}) 夜晚({count[5]})"

