from numpy.random import choice
from dataset import tiny_caltech35
import torchvision.transforms as transforms
from torchvision import models
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
# if using other models, change the imported model
from model import base_model
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import random

# if using cpu, change 'cuda' to 'cpu'
device = torch.device('cuda')


def main(config):
    transform_train = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(config.image_size, interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    train_dataset = tiny_caltech35(
        transform=transform_train, used_data=['train'])
    # if you want to add the addition set and validation set to train
    # train_dataset = tiny_caltech35(transform=transform_train, used_data=['train', 'val', 'addition'])

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=True, drop_last=True)

    val_dataset = tiny_caltech35(transform=transform_test, used_data=['val'])
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=False, drop_last=False)

    test_dataset = tiny_caltech35(transform=transform_test, used_data=['test'])
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, pin_memory=True, shuffle=False, drop_last=False)

    model = base_model(class_num=config.class_num)
    # change models
    # model = models.alexnet()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.milestones, gamma=0.1, last_epoch=-1)
    # change loss function
    creiteron = torch.nn.CrossEntropyLoss()

    # you may need train_numbers and train_losses to visualize something
    train_numbers, train_losses, train_accuracy = train(
        config, train_loader, model, optimizer, scheduler, creiteron)

    # loss and accuracy visualize
    draw(train_numbers, train_losses, 'log',
         'train numbers', 'loss', 'loss.pdf')
    draw(train_numbers, train_accuracy, 'linear',
         'train numbers', 'accuracy', 'accuracy.pdf')

    # you can use validation dataset to adjust hyper-parameters
    val_accuracy = test(val_loader, model)
    test_accuracy = test(test_loader, model)
    print('===========================')
    print("val accuracy:{}%".format(val_accuracy * 100))
    print("test accuracy:{}%".format(test_accuracy * 100))
    return test_accuracy


def train(config, data_loader, model, optimizer, scheduler, creiteron):
    model.train()
    train_losses = []
    train_accuracy = []
    train_numbers = []
    counter = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.to(device)
            label = label.to(device)
            output, last_layer = model(data)

            # introduce noise
            if config.noise != 0:
                noise_select = int(1/config.noise)
                for i in range(label.numel()):
                    if random.randint(0, 10000) % noise_select == 0:
                        label[i] = random.randint(0, 34)

            # adjust for MSELoss
            # output, idx = output.max(1)
            # output = output.float()
            # label = label.float()

            loss = creiteron(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter += data.shape[0]
            accuracy = (label == output.argmax(dim=1)).sum() * \
                1.0 / output.shape[0]

            if batch_idx % 20 == 0:
                print('Train Epoch: {} / {} [{}/{} ({:.0f}%)] Loss: {:.6f} Accuracy: {:.6f}'.format(
                    epoch, config.epochs, batch_idx *
                    len(data), len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), loss.item(), accuracy.item()))
                train_losses.append(loss.item())
                train_accuracy.append(accuracy.item())
                train_numbers.append(counter)
        scheduler.step()
        torch.save(model.state_dict(), './model.pth')
    return train_numbers, train_losses, train_accuracy


def test(data_loader, model):
    model.eval()
    correct = 0
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    plt.figure()
    xmin = xmax = ymin = ymax = 0
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            label = label.to(device)
            output, last_layer = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == label).sum()

    # tsne visualize
    #         low_dim_embs = tsne.fit_transform(
    #             last_layer.cpu().data.numpy()[:plot_only, :])
    #         labels = label.cpu().numpy()[:plot_only]
    #         X, Y = low_dim_embs[:, 0], low_dim_embs[:, 1]
    #         xmin = min(xmin, X.min())
    #         xmax = max(xmax, X.max())
    #         ymin = min(ymin, Y.min())
    #         ymax = max(ymax, Y.max())
    #         for x, y, s in zip(X, Y, labels):
    #             c = cm.rainbow(int(255 * s / 35))
    #             plt.text(x, y, s, color=c, fontdict={
    #                 'weight': 'bold', 'size': 9})
    # plt.xlim(xmin, xmax)
    # plt.ylim(ymin, ymax)
    # plt.title('t-SNE Visualize')
    # plt.show()
    accuracy = correct * 1.0 / len(data_loader.dataset)
    return accuracy


# visualize function
def draw(x_data, y_data, y_scale, x_label, y_label, filename):
    plt.figure()
    plt.axes(yscale=y_scale)
    plt.plot(x_data, y_data)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int,
                        nargs='+', default=[112, 112])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--class_num', type=int, default=35)
    parser.add_argument('--learning_rate', type=float, default=0.0035)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--milestones', type=int,
                        nargs='+', default=[40, 50])
    # add noise rate argument
    parser.add_argument('--noise', type=int, default=0)

    config = parser.parse_args()
    main(config)

    # print(torch.cuda.is_available())

    # learning rate test

    # learning_rate = [0.01, 0.008, 0.006, 0.005, 0.004,
    #                  0.003, 0.002, 0.001, 0.0008, 0.0005, 0.0002, 0.0001]
    # accuracy = []
    # for rate in learning_rate:
    #     config.learning_rate = rate
    #     accuracy.append(main(config))
    # draw(learning_rate, accuracy, 'linear',
    #      'learning rate', 'accuracy', 'rate.pdf')

    # noise rate test

    # noise_rate = [0.1, 0.2, 0.25, 0.33, 0.5, 1]
    # accuracy = []
    # for rate in noise_rate:
    #     config.noise = rate
    #     accuracy.append(main(config))
    # draw(noise_rate, accuracy, 'linear',
    #      'noise rate', 'accuracy', 'rate.pdf')
