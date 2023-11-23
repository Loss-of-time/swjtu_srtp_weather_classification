from typing import Any
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
            "q": 0,  # 晴
            "e": 1,  # 阴
            "r": 2,  # 雨
            "a": 3,  # 雪
            "s": 4,  # 雾
            "f": 5,  # 夜晚
        }
        self.set = []

        for root, dirs, files in os.walk(self.root):
            target = root.split("\\")[-1]
            print(root, target)
            for file in files:
                type_ = file.split(".")[-2]
                t = self.dict[type_]
                if type_ != "d":
                    pic = read_image(os.path.join(root, file))
                    pic = transform(pic)
                    information = {"image": pic, "target": t}
                    self.set.append(information)

    def __getitem__(self, index):
        img, target = self.set[index]["image"], self.set[index]["target"]
        return img, target

    def __len__(self):
        return len(self.set)

    def __str__(self):
        count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        for item in self.set:
            target = item["target"]
            count[target] += 1
        return f"图片数量：晴({count[0]}) 阴({count[1]}) 雨({count[2]}) 雪({count[3]}) 雾({count[4]}) 夜晚({count[5]})"


from pathlib import Path
from torchvision.transforms import Compose


class RSCM2017(Dataset):
    type_num: int = 6
    type_dict = {
        "cloudy": 0,
        "haze": 1,
        "rainy": 2,
        "snow": 3,
        "sunny": 4,
        "thunder": 5,
    }

    def __init__(self, path: str | Path, transform: Compose) -> None:
        super().__init__()
        self.images = []
        self.labels = []

        for root, dirs, files in os.walk(path):
            for file in files:
                if file.split(".")[-1] == "jpg":
                    pic = read_image(os.path.join(root, file))
                    if pic.shape[0] == 1:
                        pic = pic.repeat(3, 1, 1)
                    pic = transform(pic)
                    self.images.append(pic)
                    self.labels.append(self.type_dict[root.split("\\")[-1]])

    def __getitem__(self, index) -> Any:
        image = self.images[index]
        label = self.labels[index]
        return image, label

    def __len__(self) -> int:
        return len(self.images)


from torchvision import transforms

if __name__ == "__main__":
    transform = Compose(
        [
            transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(90),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    path = Path(
        r"C:\Code\Python\swjtu_srtp_weather_classification\data\weather_classification"
    )

    data = RSCM2017(path, transform)

    print(len(data))
