# local
from utils import plot_curve, delete_folder_contents
from dataset import cls_dataset
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
from torch.utils.data import DataLoader
from loguru import logger


# ------------------------------------------------------------
# Function
# ------------------------------------------------------------


class TorchFactory(object):
    @staticmethod
    def get_dataloader(data_name: str, transform: transforms.Compose):
        features = 0
        trainset = cls_dataset(train_set_path, transform=transform)
        testset = cls_dataset(test_set_path, transform=transform)
        if data_name == 'mine':
            features = 7
        elif data_name == 'cifar10':
            trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
            features = 10
        train_dataloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(
            testset, batch_size=batch_size, shuffle=True)
        return train_dataloader, test_dataloader, features

    @staticmethod
    def get_model(model_name: str, new_features: int) -> nn.Module:
        pretrained_dict = {
            'resnet18': models.resnet18,
            'resnet50': models.resnet50,
            'vgg16':  models.vgg16,
            'alexnet': models.alexnet
        }
        if model_name in pretrained_dict:
            model = pretrained_dict[model_name](pretrained=True)
            if 'resnet' in model_name:
                model.fc = nn.Linear(model.fc.in_features, new_features)
            elif 'vgg' in model_name:
                num_features = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(
                    num_features, new_features)
            elif 'alexnet' in model_name:
                num_features = model.classifier[6].in_features
                model.classifier[6] = torch.nn.Linear(
                    num_features, new_features)
            return model
        else:
            return TorchFactory.get_model('resnet18', new_features)

    @staticmethod
    def get_optimizer(optimizer_name: str, model: nn.Module) -> optim.Optimizer:
        optimizer_name = optimizer_name.lower()
        if optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learn_rate)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learn_rate,
                                  momentum=0.9, weight_decay=5e-4)
        else:
            optimizer = TorchFactory.get_optimizer('adam', model)
        return optimizer


class TrainInfo(object):
    loss_list = []
    train_acc_list = []
    test_acc_list = []

    def __init__(self, loss: list, train: list, test: list) -> None:
        self.loss_list = loss
        self.train_acc_list = train
        self.test_acc_list = test

    def best_index(self):
        return self.test_acc_list.index(max(self.test_acc_list))





def judge(model: nn.Module, dataloader: DataLoader) -> float:
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in dataloader:
            x, label = x.to(device), label.to(device)
            logits = model.forward(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
        acc = total_correct / total_num
    return acc


def get_model_name(epoch: int, name: str = 'resnet', suffix='pth') -> str:
    return '{}_epoch{:03d}.{}'.format(name, epoch, suffix)


def train(epoch: int, model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, criteon: nn.modules.loss._WeightedLoss, optimizer: optim.Optimizer, model_name : str, features:int, **kwargs) -> TrainInfo:
    def train_epoch():
        model.train()
        for batchidx, (x, label) in enumerate(train_dataloader):
            x, label = x.to(device), label.to(device)
            logits = model.forward(x)
            loss = criteon.forward(logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
        return loss
    loss_list = []
    train_acc_list = []
    test_acc_list = []
    for ep in range(1, epoch + 1):
        # train
        model.train()
        start = time.time()
        loss = train_epoch()
        logger.info(f'Epoch {ep}')
        logger.info(f'loss: {loss}')
        logger.info(f'lr_rate: {optimizer.param_groups[0]["lr"]}')
        loss_list.append(float(loss))

        acc = judge(model, train_dataloader)
        logger.info(f"trian set acc: {acc:.2f}")
        train_acc_list.append(float(acc))

        acc = judge(model, test_dataloader)
        logger.info(f"test set acc: {acc:.2f}")
        test_acc_list.append(float(acc))

        torch.save(model.state_dict(), model_save_path /
                   get_model_name(ep, model_name))

        end = time.time()
        logger.info(f"epoch consume: {end - start:.2f}s")
    return TrainInfo(loss_list, train_acc_list, test_acc_list)


# ------------------------------------------------------------
# Init
# ------------------------------------------------------------
logger.add(log_path, rotation=log_rotation)
device = torch.device('cuda')
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

train_dataloader, test_dataloader, features = TorchFactory.get_dataloader(data_name, transform)
# -----------------------------
model = TorchFactory.get_model(model_name, features)
model.to(device)
logger.debug(model)
# -----------------------------
criteon = nn.CrossEntropyLoss().to(device)
optimizer = TorchFactory.get_optimizer(optimizer_name, model)
# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
delete_folder_contents(model_save_path)
info = train(
    epoch,
    model,
    train_dataloader,
    test_dataloader,
    criteon,
    optimizer,
    model_name,
    features
)
# -----------------------------
best_index = info.best_index()
best_model_name = get_model_name(best_index, model_name)
best_model_path = model_save_path / best_model_name
shutil.copy(best_model_path, best_model_save_path)
logger.info('The best model is {}, which accuracy is {:.2f}. And save in {}'.format(
    best_model_name, info.test_acc_list[best_index], best_model_save_path))

plot_curve(info.loss_list)
plot_curve(info.train_acc_list)
plot_curve(info.test_acc_list)
