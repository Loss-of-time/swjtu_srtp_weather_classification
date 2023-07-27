import time
import json
from pathlib import Path

from utils import plot_curve
from dataset import cls_dataset
from settings import train_set_path, test_set_path

import torch
from torch import nn, optim
import torchvision.models as models

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
# ------------------------------------------------------------
batch_size = 40
epoch = 32
learn_rate = 1e-3
# data = Path('./data')
save_path = Path('./models')
device = torch.device('cuda')

loss_list = []
acc_list = []

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# data = str(data)
# trainset = datasets.CIFAR10(
#     root=data, train=True, download=True, transform=transform1)
# testset = datasets.CIFAR10(root=data, train=False,
#    download=True, transform=transform2)
trainset = cls_dataset(train_set_path, transform=transform)
testset = cls_dataset(test_set_path, transform=transform)

train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

Model = models.resnet18(pretrained=True)
# Model.fc = nn.Linear(Model.fc.in_features, 10)
Model.fc = nn.Linear(Model.fc.in_features, 7)
Model.to(device)

criteon = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(Model.parameters(), lr=learn_rate)
# optimizer = optim.SGD(Model.parameters(), lr=learn_rate,
                    #   momentum=0.9, weight_decay=5e-4)
# scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch / 3 + 1))
# ------------------------------------------------------------
print(Model)
for epoch in range(1, epoch + 1):
    # train
    Model.train()
    start = time.time()
    for batchidx, (x, label) in enumerate(train_dataloader):
        x, label = x.to(device), label.to(device)
        logits = Model(x)
        loss = criteon(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
    print(f'Epoch {epoch}')
    print(f'loss: {loss}')
    print(f'lr_rate: {optimizer.param_groups[0]["lr"]}')
    Model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in test_dataloader:
            x, label = x.to(device), label.to(device)
            logits = Model(x)
            pred = logits.argmax(dim=1)
            total_correct += torch.eq(pred, label).float().sum().item()
            total_num += x.size(0)
        acc = total_correct / total_num
        print(f"acc: {acc}")
    loss_list.append(float(loss))
    acc_list.append(float(acc))
    name = 'resnet50'
    suffix = 'pth'
    torch.save(Model.state_dict(), save_path / f'{name}.{suffix}')
    end = time.time()
    print(f"epoch consume: {end - start}")

plot_curve(loss_list)
plot_curve(acc_list)
