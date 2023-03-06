from time import time

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchvision import datasets
from torchvision import transforms
import multiprocessing as mp
import os


class AlexNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_loader: DataLoader,
            criterion,
            optimizer: torch.optim.Optimizer,
            test_loader: DataLoader):
        self.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
        self.global_rank = int(os.environ["SLURM_PROCID"])
        self.local_rank = self.global_rank % self.gpus_per_node

        # Transfer model to GPU specified by local rank
        self.model = model.to(self.local_rank)
        # Wrap model using DDP
        self.model = DDP(self.model, device_ids=[self.local_rank])

        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.test_loader = test_loader

    def _run_batch(self, inputs, targets):
        self.optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_loader))[0])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_loader)}")
        # Necessary to make shuffling work properly across multiple epochs.
        # Otherwise, the same ordering will always be used.
        self.train_loader.sampler.set_epoch(epoch)
        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.local_rank)
            targets = targets.to(self.local_rank)
            self._run_batch(inputs, targets)

    def _store_model(self):
        path = "model.pt"
        torch.save(self.model.module.state_dict(), path)
        print(f"Stored model at {path}")

    def train(self, max_epochs: int, store_model=False):
        start = time()
        self.model.train()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)

        end = time()
        print(f"Training on {self.global_rank} with {max_epochs} epochs took {end - start} seconds")
        if store_model and self.global_rank == 0:
            self._store_model()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in self.test_loader:
                images = images.to(self.local_rank)
                labels = labels.to(self.local_rank)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            # print(f"Testing on rank {self.global_rank}:"
            #      f"\n Accuracy of the network on my {total} test images: {100 * correct / total}")

        training_summary = torch.IntTensor([correct, total])
        training_summary = training_summary.to(self.local_rank)
        dist.all_reduce(training_summary, op=dist.ReduceOp.SUM, group=dist.group.WORLD)
        if self.global_rank == 0:
            correct, total = training_summary[0], training_summary[1]
            print(f"Accuracy of the network on {total} test images: {100 * correct / total}")


def ddp_setup():
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NPROCS"])

    addr = os.environ["SLURM_LAUNCH_NODE_IPADDR"]
    port = "29500"
    os.environ["MASTER_PORT"] = port
    os.environ["MASTER_ADDR"] = addr

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def load_train_objs(learning_rate: float):
    model = AlexNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)
    return model, criterion, optimizer


def create_data_loader_cifar10(batch_size: int):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.247, 0.243, 0.261]
    )

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize
    ])

    # TODO: Find a better way to set num_workers
    num_workers = 4
    # Load training data
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=num_workers, pin_memory=True)
    # Load test data
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_sampler = DistributedSampler(dataset=testset, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, sampler=test_sampler,
                                              num_workers=num_workers)
    return train_loader, test_loader


def main(total_epochs: int, batch_size: int, learning_rate: float):
    ddp_setup()
    model, criterion, optimizer = load_train_objs(learning_rate)
    train_loader, test_loader = create_data_loader_cifar10(batch_size)
    trainer = Trainer(model, train_loader, criterion, optimizer, test_loader)
    trainer.train(total_epochs, store_model=False)
    trainer.test()

    # TODO: Check if I can leave it in later
    # dist.destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size on each device (default: 64)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate (default 0.005)')
    args = parser.parse_args()

    main(args.total_epochs, args.batch_size, args.lr)
