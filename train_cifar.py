import sys
sys.path.append('./python')
sys.path.append('./apps')
import needle as ndl
from models import ResNet9
from simple_ml import train_cifar10, evaluate_cifar10

device = ndl.cpu()
train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
test_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=False)
test_dataloader = ndl.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)
model = ResNet9(device=device, dtype="float32")
train_acc, train_loss = train_cifar10(model, train_dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
      lr=0.001, weight_decay=0.001, device=device)
test_acc, test_loss = evaluate_cifar10(model, test_dataloader, device=device)
print(f'train_acc={train_acc}, train_loss={train_loss}, test_acc={test_acc}, test_loss={test_loss}')
# cpu: train_acc=0.69274, train_loss=0.8652232885360718, test_acc=0.4706, test_loss=1.8050200939178467
# cuda: train_acc=0.70006, train_loss=0.8554841876029968, test_acc=0.3841, test_loss=2.476391077041626