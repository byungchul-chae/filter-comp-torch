import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# 데이터셋에 대한 전처리 정의
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# CIFAR-10 훈련 및 테스트 데이터셋 로드
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                           shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64,
                                          shuffle=False, num_workers=2)

# ResNet-18 모델 로드 및 마지막 레이어 교체
resnet18 = models.resnet18(pretrained=True)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 10)  # CIFAR-10은 10개의 클래스를 가짐

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet18 = resnet18.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)


# 훈련 함수
def train(model, device, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 200:.3f}')
                running_loss = 0.0


# 훈련 실행
train(resnet18, device, train_loader, criterion, optimizer, num_epochs=2)


# 추론 및 정확도 계산 함수
def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the 10000 test images: {100 * correct / total}%')


# 추론 실행
test(resnet18, device, test_loader)

import torchvision.models as models

# 사전 훈련된 ResNet-18 모델 로드
resnet18 = models.resnet18(pretrained=True)

# 모델의 모든 컨볼루션 레이어를 순회하며 3x3 필터를 사용하는 레이어의 파라미터 확인
for name, module in resnet18.named_modules():
    if isinstance(module, torch.nn.Conv2d) and module.kernel_size == (3, 3):
        print(f"Layer: {name}")
        print(f"Parameters: {list(module.parameters())}")
        print("-----")
