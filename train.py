# local
from utils import plot_curve, delete_folder_contents
from dataset import cls_dataset, RSCM2017
from settings import *

# official
import json
import os
import shutil
import time

# third party
import torch
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from loguru import logger


# ------------------------------------------------------------
# Function
# ------------------------------------------------------------


class TorchFactory(object):
    """
    A factory class for creating PyTorch models, dataloaders, and optimizers.
    """

    @staticmethod
    def get_dataloader(data_name: str, transform: transforms.Compose):
        """
        Get the dataloader for the specified dataset.

        Args:
            data_name (str): The name of the dataset.
            transform (torchvision.transforms.Compose): The data transformation pipeline.

        Returns:
            tuple: A tuple containing the train dataloader, test dataloader, and number of features.
        """
        features = 0
        if data_name == "mine":
            train_set = cls_dataset(train_set_path, transform=transform)
            test_set = cls_dataset(test_set_path, transform=transform)
            logger.info(train_set)
            features = cls_dataset.type_num
        elif data_name == "rscm":
            path = Path(
                r"C:\Code\Python\swjtu_srtp_weather_classification\data\weather_classification"
            )
            date_set = RSCM2017(path, transform)
            train_set, test_set = random_split(date_set, [0.8, 0.2])
            features = RSCM2017.type_num
        elif data_name == "cifar10":
            train_set = datasets.CIFAR10(
                root="./data/cifar10", train=True, download=True, transform=transform
            )
            test_set = datasets.CIFAR10(
                root="./data/cifar10", train=False, download=True, transform=transform
            )
            features = 10
        else:
            return TorchFactory.get_dataloader("mine", transform)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        return train_dataloader, test_dataloader, features

    @staticmethod
    def get_model(model_name: str, new_features: int) -> nn.Module:
        """
        Get the specified PyTorch model.

        Args:
            model_name (str): The name of the model.
            new_features (int): The number of output features for the last layer.

        Returns:
            torch.nn.Module: The PyTorch model.
        """
        pretrained_dict = {
            "resnet18": models.resnet18,
            "resnet50": models.resnet50,
            "resnet152": models.resnet152,
            "vgg16": models.vgg16,
            "alexnet": models.alexnet,
        }
        if model_name in pretrained_dict:
            model = pretrained_dict[model_name](pretrained=True)
            if "resnet" in model_name:
                model.fc = nn.Linear(model.fc.in_features, new_features)
            elif "vgg" in model_name:
                num_features = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(num_features, new_features)
            elif "alexnet" in model_name:
                num_features = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(num_features, new_features)
            return model
        else:
            return TorchFactory.get_model("resnet18", new_features)

    @staticmethod
    def get_optimizer(optimizer_name: str, model: nn.Module) -> optim.Optimizer:
        """
        Get the specified optimizer for the given model.

        Args:
            optimizer_name (str): The name of the optimizer.
            model (torch.nn.Module): The PyTorch model.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        optimizer_name = optimizer_name.lower()
        if optimizer_name == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        elif optimizer_name == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=5e-4
            )
        elif optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(model.parameters(), lr=learn_rate)
        else:
            optimizer = TorchFactory.get_optimizer("adam", model)
        return optimizer


def judge(
    model: nn.Module, dataloader: DataLoader, features: int
) -> tuple[float, list, list]:
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader for the dataset.
        features (int): The number of features/classes in the dataset.

    Returns:
        tuple[float, list, list]: A tuple containing the accuracy, the number of correct predictions for each class,
        and the total number of instances for each class.
    """
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        each_correct = [0 for i in range(features)]
        each_total = [0 for i in range(features)]
        for x, label in dataloader:
            x, label = x.to(device), label.to(device)
            logits = model.forward(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
            # 召回率统计
            for idx, _label in enumerate(label):
                each_total[_label] += 1
                if pred[idx] == _label:
                    each_correct[_label] += 1
        acc = total_correct / total_num
    return acc, each_correct, each_total


def get_model_name(epoch: int, name: str = "resnet", suffix="pth") -> str:
    return "{}_epoch{:03d}.{}".format(name, epoch, suffix)


def train(
    epoch: int,
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criteon: nn.modules.loss._WeightedLoss,
    optimizer: optim.Optimizer,
    model_name: str,
    features: int,
) -> tuple[list, list, list]:
    """
    Trains the model for the specified number of epochs.

    Args:
        epoch (int): The number of epochs to train the model.
        model (nn.Module): The model to be trained.
        train_dataloader (DataLoader): The dataloader for the training data.
        test_dataloader (DataLoader): The dataloader for the test data.
        criteon (nn.modules.loss._WeightedLoss): The loss function.
        optimizer (optim.Optimizer): The optimizer used for training.
        model_name (str): The name of the model.
        features (int): The number of features in the data.
        **kwargs: Additional keyword arguments.

    Returns:
        tuple[list, list, list]: A tuple containing the lists of loss, train accuracy, and test accuracy.
    """

    def train_epoch():
        model.train()
        loss_max = 0.0
        for batch_idx, (x, label) in enumerate(train_dataloader):
            x, label = x.to(device), label.to(device)
            logits = model.forward(x)
            loss = criteon.forward(logits, label)
            loss_max = max(loss_max, loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        return loss_max

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for ep in range(1, epoch + 1):
        # train
        start = time.time()
        loss = train_epoch()
        logger.info(f"Epoch {ep}")
        logger.info(f"loss: {loss}")
        logger.info(f'lr_rate: {optimizer.param_groups[0]["lr"]}')
        loss_list.append(float(loss))

        acc, correct, total = judge(model, train_dataloader, features)
        logger.info(f"train set acc: {acc:.2f}")
        train_acc_list.append(float(acc))

        acc, correct, total = judge(model, test_dataloader, features)
        logger.info(f"test set acc: {acc:.2f}")
        recall = ""
        for i in range(features):
            recall += f"{correct[i]} / {total[i]}   "
        logger.info(f"RECALL: {recall}")
        test_acc_list.append(float(acc))

        torch.save(model.state_dict(), model_save_path / get_model_name(ep, model_name))

        end = time.time()
        logger.info(f"epoch consume: {end - start:.2f}s")
    return loss_list, train_acc_list, test_acc_list


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Init
    # ------------------------------------------------------------
    logger.add(log_path, rotation=log_rotation)
    device = torch.device("cuda")
    # -----------------------------
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(90),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    logger.info("Start training...")
    start_time = time.time()
    train_dataloader, test_dataloader, features = TorchFactory.get_dataloader(
        data_name, transform
    )
    end_time = time.time()
    logger.info(f"Data loading consume: {end_time - start_time:.2f}s")
    # -----------------------------
    start_time = time.time()
    model = TorchFactory.get_model(model_name, features)
    model.to(device)
    end_time = time.time()
    logger.info(f"Model loading consume: {end_time - start_time:.2f}s")
    # logger.debug(model)
    # -----------------------------
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = TorchFactory.get_optimizer(optimizer_name, model)
    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------
    delete_folder_contents(model_save_path)
    loss_list, train_acc_list, test_acc_list = train(
        epoch,
        model,
        train_dataloader,
        test_dataloader,
        criteon,
        optimizer,
        model_name,
        features,
    )
    best_index = test_acc_list.index(max(test_acc_list))
    # -----------------------------
    best_model_name = get_model_name(best_index, model_name)
    best_model_path = model_save_path / best_model_name
    shutil.copy(best_model_path, best_model_save_path)
    # 虽然保存了表现最好的模型，但我们不适用它进行在训练，因为它可能会过拟合
    logger.info(
        "The best model is {}, which accuracy is {:.2f}. And save in {}".format(
            best_model_name, test_acc_list[best_index], best_model_save_path
        )
    )

    plot_curve(loss_list)
    plot_curve(train_acc_list)
    plot_curve(test_acc_list)
