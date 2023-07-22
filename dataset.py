from torch.utils.data import Dataset
from torchvision.io import read_image
import os

# 从一个文件夹中创建数据集，文件格式{name}.{label}.{suffix} （使用.分隔文件名，类型和扩展名）
class cls_dataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()
        self.root = root
        self.set = []

        for root, dirs, files in os.walk(self.root):
            target = root.split('\\')[-1]
            print(root, target)
            for file in files:
                type_ = file.split('.')[-2]
                if (type_ != 'd'):
                    pic = read_image(os.path.join(root, file))
                    pic = transform(pic)
                    t = int(type_)
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
