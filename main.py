import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class WeightManager:
    def __init__(self, filepath):
        self.filepath = filepath

    def save_weights(self, model):
        torch.save(model.state_dict(), self.filepath)
        print(f"Saved weights to {self.filepath}")

    def load_weights(self, model):
        model.load_state_dict(torch.load(self.filepath))
        print(f"Loaded weights from {self.filepath}")

def main():
    # 데이터셋 준비
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST('./data', download=True, train=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 테스트 데이터셋 로딩
    test_dataset = datasets.MNIST('./data', download=True, train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 모델, 손실 함수, 최적화 알고리즘 초기화
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 학습
    for epoch in range(3):  # 간단한 예제를 위해 1 에폭만 실행
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Completed")

    # 가중치 저장
    weight_manager = WeightManager('./model_weights.pth')
    weight_manager.save_weights(model)

    # 가중치 불러오기 및 추론 준비
    model = SimpleCNN()  # 새 모델 인스턴스 생성
    weight_manager.load_weights(model)

    def test(model, device, test_loader):
        model.eval()  # 모델을 추론 모드로 설정
        correct = 0
        total = 0
        with torch.no_grad():  # 추론 시에는 기울기를 계산할 필요가 없습니다
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total}%')

    # 모델을 CPU나 GPU에 배치 (사용 가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 테스트 함수 호출
    test(model, device, test_loader)

if __name__ == "__main__":
    main()
