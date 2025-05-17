import torch
from torch import nn
from torch.autograd import Variable
import random

import torchvision.transforms as tfs
from torchvision import utils
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision.datasets import MNIST

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from IPython.display import HTML
from IPython.display import display

import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk, Image

SEED = 1392
torch.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

NOISE_DIM = 96
NUM_EPOCHS = 20
batch_size = 128

bce_loss = nn.BCEWithLogitsLoss()

# def show_images(images): # 定义画图工具
#     images = deprocess_img(images)
#     images = images.detach().cpu().numpy()
#     # images = np.transpose(images, (0, 2, 3, 1))
#     images = np.concatenate(images, axis=1)
#     # plt.ion()
#     plt.imshow(images)
#     plt.axis('off')
#     plt.show()
    # plt.close()
def show_images(img_list):
    img_list = [deprocess_img(img) for img in img_list]
    fig = plt.figure(figsize=(10, 5))
    plt.axis("off")
    ims = [[plt.imshow(item.permute(1, 2, 0), animated=True)] for item in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    ani.save("image_change.gif", writer='pillow')

    ani.save('image_change.mp4', writer='ffmpeg', fps=1000 / 50)
    display(ani)
    # HTML(ani.to_jshtml())

# 归一化
def preprocess_img(x):
    x = tfs.ToTensor()(x)
    x = tfs.Resize(32)(x)
    return (x - 0.5) / 0.5

def deprocess_img(x):
    # x = x.detach().cpu().numpy()
    return (x + 1.0) / 2.0

#训练集
# 加载保存的模型或变量
# loaded_data = torch.load(r"D:\new_program\pythonProject\pytorchUse\DeepLearning\GAN\mnist\MNIST\processed\training.pt")
# print(loaded_data)

# 判别器
def discriminator():
    # net = nn.Sequential(
    #     nn.Linear(784, 256),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(256, 1),
    #     nn.Sigmoid()
    # )
    # net = nn.Sequential(
    #     nn.Linear(784, 512),  # 输入特征数为784，输出为512
    #     nn.BatchNorm1d(512),
    #     nn.LeakyReLU(0.2),  # 进行非线性映射
    #     nn.Linear(512, 256),  # 进行一个线性映射
    #     nn.BatchNorm1d(256),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(256, 1),
    #     nn.Sigmoid()  # 也是一个激活函数，二分类问题中，
    #     # sigmoid可以班实数映射到【0,1】，作为概率值，
    #     # 多分类用softmax函数
    # )
    net = nn.Sequential(
        # state size. (64*2) x 16 x 16
        nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (64*4) x 8 x 8
        nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (64*8) x 4 x 4
        nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
        # state size. (1) x 1 x 1
        nn.Sigmoid()
    )
    return net

#生成器
def generator(noise_dim=NOISE_DIM):
    # net = nn.Sequential(
    #     nn.Linear(noise_dim, 128),
    #     nn.ReLU(),
    #     nn.Linear(128, 784),
    #     nn.Tanh()
    # )
    # net = nn.Sequential(
    #     nn.Linear(NOISE_DIM, 128),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(128, 256),
    #     nn.BatchNorm1d(256),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(256, 512),
    #     nn.BatchNorm1d(512),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(512, 1024),
    #     nn.BatchNorm1d(1024),
    #     nn.LeakyReLU(0.2),
    #     nn.Linear(1024, 784),
    #     nn.Tanh()
    # )
    net = nn.Sequential(
        # state size. (ngf*8) x 4 x 4
        nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.ReLU(True),
        # state size. (ngf*4) x 8 x 8
        nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 2),
        nn.ReLU(True),
        # state size. (ngf*2) x 16 x 16
        nn.ConvTranspose2d(64 * 2, 1, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. (nc) x 32 x 32
    )
    return net

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.image = nn.Sequential(
            # input is (1) x 32 x 32
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (64) x 16 x 16
        )
        self.label = nn.Sequential(
            # input is (num_classes) x 32 x 32
            nn.Conv2d(num_classes, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            # state size. (64) x 16 x 16
        )
        self.main = discriminator()
    def forward(self, image, label):
        image = self.image(image)
        label = self.label(label)
        # print(image.shape)
        # print(label.shape)
        incat = torch.cat((image, label), dim=1)
        # print(incat.shape)
        return self.main(incat)

class Generator(nn.Module):
    def __init__(self, num_classes):
        super(Generator, self).__init__()
        self.image = nn.Sequential(
            # state size. (nz) x 1 x 1
            nn.ConvTranspose2d(NOISE_DIM, 64 * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True)
            # state size. (ngf*4) x 4 x 4
        )
        self.label = nn.Sequential(
            # state size. (num_classes) x 1 x 1
            nn.ConvTranspose2d(num_classes, 64 * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True)
            # state size. (ngf*4) x 4 x 4
        )
        self.main = generator()

    def forward(self, image, label):
        # print(image.shape)
        image = self.image(image)
        label = self.label(label)
        incat = torch.cat((image, label), dim=1)
        return self.main(incat)

# 判别器损失
def discriminator_loss(logits_real, logits_fake):
    loss_real = bce_loss(logits_real, torch.ones_like(logits_real))
    # 真图的分数，越大越好
    loss_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
    # 假图的分数，越小越好
    return loss_real + loss_fake

# 生成器损失
def generator_loss(logits_fake):
    # G想骗过D，故让其越接近1越好
    return bce_loss(logits_fake, torch.ones_like(logits_fake))

# 优化器
def get_optimizer(net):
    # 将网络的参数传递给优化器,学习率为 2e-4,betas 设置为 (0.5, 0.999)
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.5, 0.999))
    return optimizer

# 训练过程
def train_a_gan(train_data, D_net, G_net, D_optimizer, G_optimizer, discriminator_loss, generator_loss,
                label_onehots, label_fills, noise_size=NOISE_DIM, num_epochs=NUM_EPOCHS):
    img_list = []
    G_losses = []
    D_losses = []
    for epoch in range(num_epochs):
        loss_G_single = []
        loss_D_single = []
        for i, (img, label) in enumerate(train_data):
            # 图片和标签
            # real_data = img.view(img.size(0), -1)
            # real_label = label.view(label.size(0), -1)
            # print(real_data.size())
            # print(real_label.size())
            # torch.Size([128, 784])
            # torch.Size([128, 1])
            real_data = img
            real_data = Variable(real_data).cuda()
            # real_label = Variable(real_label).cuda()

            # 使用onehot表示图像的标签（10个），数据shape见下面
            G_label = label_onehots[label]
            D_label = label_fills[label]
            # print(G_label.shape) # torch.Size([128, 10, 1, 1])
            # print(D_label.shape) # torch.Size([128, 10, 28, 28])

            # 训练判别器
            # noise = Variable(torch.randn(img.size(0), noise_size)).cuda()
            # 生成fake图片
            noise = Variable(torch.randn(img.size(0), noise_size, 1, 1)).cuda()
            # print(noise.shape) # torch.Size([128, 96])
            fake_data = G_net(noise, G_label)

            # 分别输出D对真实和虚假图片给出的logits，偏向1则真，偏向0则假
            logits_real = D_net(real_data, D_label).view(-1)
            logits_fake = D_net(fake_data, D_label).view(-1)
            # 见discriminator_loss函数
            d_loss = discriminator_loss(logits_real, logits_fake)
            # 更新D优化器
            D_optimizer.zero_grad()
            d_loss.backward()
            D_optimizer.step()

            # 训练生成器
            noise = Variable(torch.randn(img.size(0), noise_size, 1, 1)).cuda()
            fake_data = G_net(noise, G_label)
            # print(fake_data.shape) # torch.Size([128, 784]) # 之前的GAN网络结构结果
            logits_fake = D_net(fake_data, D_label).view(-1)
            g_loss = generator_loss(logits_fake)
            # 更新G优化器
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()

            loss_G_single.append(g_loss.mean().item())
            loss_D_single.append(d_loss.mean().item())

            # if i == 546:
            if i % 100 == 0:
            # if epoch == NUM_EPOCHS - 1 and i==469:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}'.format(epoch, num_epochs, i, len(train_data), d_loss.item(), g_loss.item()))
        # 测试一下G
        noise = Variable(torch.randn(50, noise_size, 1, 1)).cuda()
        fixed_label = label_onehots[torch.arange(10).repeat(5).sort().values]
        fake_data = G_net(noise, fixed_label)
        # print(fake_data.shape) # torch.Size([64, 784]) # torch.Size([50, 1, 32, 32])
        # show_images(utils.make_grid(fake_data, nrow=10)) # fake_data

        # img_list列表用于后续用动画演示训练过程
        img_list.append(utils.make_grid(fake_data.detach().cpu(), nrow=10))

        loss_G_single = np.array(loss_G_single).mean()
        loss_D_single = np.array(loss_D_single).mean()
        # print(loss_G_single, loss_D_single)
        G_losses.append(loss_G_single)
        D_losses.append(loss_D_single)
    show_images(img_list)
    # # 保存模型
    # torch.save(G_net.state_dict(), "generator.pt")
    # torch.save(G_net.state_dict(), "discriminator.pt")

    plt.figure(figsize=(20, 10))
    plt.title("Generator and Discriminator Loss Change")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# ------------------------------以上完全是按照原本的函数结构，以下是新的函数---------------------------------

def draw_real_images(dataset, label_onehots):
    imgs = {}
    # 采样100个图像
    for x, y in dataset:
        if y not in imgs:
            imgs[y] = []
        elif len(imgs[y]) != 10:
            imgs[y].append(x)
        elif sum(len(imgs[key]) for key in imgs) == 100:
            break
        else:
            continue
    # 采样100个图像并且按照数字顺序排列
    imgs = sorted(imgs.items(), key=lambda x: x[0])
    imgs = [torch.stack(item[1], dim=0) for item in imgs]
    imgs = torch.cat(imgs, dim=0)
    '''
    仅仅看原图
    '''
    plt.figure(figsize=(10, 10))
    plt.title("MNIST IMAGE EXAMPLE")
    plt.axis('off')
    imgs = utils.make_grid(imgs, nrow=10)
    plt.imshow(imgs.permute(1, 2, 0) * 0.5 + 0.5)
    # deprocess_img
    plt.show()
    '''
    对比图
    '''
    # Size of the Figure
    plt.figure(figsize=(20, 10))

    # Plot the real images
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.title("Real Images")
    imgs = utils.make_grid(imgs, nrow=10)
    plt.imshow(imgs.permute(1, 2, 0) * 0.5 + 0.5)

    # Load the Best Generative Model
    netG = Generator(num_classes=10)
    netG.load_state_dict(torch.load('generator.pt', map_location=torch.device('cpu')))
    netG.eval()

    # Generate the Fake Images
    noise = Variable(torch.randn(100, NOISE_DIM, 1, 1)).cuda()
    fixed_label = label_onehots[torch.arange(10).repeat(10).sort().values]
    with torch.no_grad():
        fake = netG(noise.cpu(), fixed_label.cpu())

    #
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    fake = utils.make_grid(fake, nrow=10)
    plt.imshow(fake.permute(1, 2, 0) * 0.5 + 0.5)

    # 保存结果
    plt.savefig('comparation.jpg', bbox_inches='tight')

def weights_init(m):
    # 所有模型权重应当从均值为0，标准差为0.02的正态分布中随机初始化。
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
def create_one_hot_labels(num_classes = 10, image_size = 32):
    """
    Creates one-hot labels for both generator and discriminator.

    Args:
      num_classes: Number of classes (10 for MNIST).
      image_size: Size of the image (28 for MNIST).

    Returns:
      Tuple: (label_1hots, label_fills).
      label_onehots (torch.Tensor): One-hot labels for generator (10 x 10 x 1 x 1).
      label_fills (torch.Tensor): One-hot labels for discriminator (10 x 10 x 28 x 28).
    """
    # G网络标签
    label_onehots = torch.zeros(num_classes, num_classes).cuda()
    for i in range(num_classes):
        label_onehots[i, i] = 1
    label_onehots = label_onehots.view(num_classes, num_classes, 1, 1).cuda()

    # D网络标签
    label_fills = torch.zeros(num_classes, num_classes, image_size, image_size).cuda()
    ones = torch.ones(image_size, image_size).cuda()
    for i in range(num_classes):
        label_fills[i][i] = ones
    label_fills = label_fills.cuda()

    return label_onehots, label_fills

def run_gui(label_onehots):
    # 创建主窗口
    root = tk.Tk()
    root.title('CDCGAN 图像生成器')

    # 创建输入框
    input_label = tk.Label(root, text='输入数字:')
    input_label.pack()

    input_entry = tk.Entry(root)
    input_entry.pack()

    # 生成图像函数
    def generate_image():
        # 从输入框中获取数字
        try:
            number = int(input_entry.get())
        except ValueError:
            messagebox.showinfo('错误','请输入数字')
            return

        # 生成图像
        # Load the Best Generative Model
        netG = Generator(num_classes=10)
        netG.load_state_dict(torch.load('generator.pt', map_location=torch.device('cpu')))
        netG.eval()
        # Generate the Fake Images
        noise = torch.randn(1, NOISE_DIM, 1, 1)
        fixed_label = label_onehots[number].unsqueeze(0)
        # print(noise.shape) # torch.Size([1, 96, 1, 1])
        # print(fixed_label.shape) # torch.Size([1, 10, 1, 1])
        with torch.no_grad():
            fake = netG(noise.cpu(), fixed_label.cpu())
        # print(fake.shape) # torch.Size([1, 1, 32, 32])


        # 将图像转换为 Tkinter 可用格式
        image = Image.fromarray(fake.numpy().squeeze() * 255).resize((256, 256))
        image_tk = ImageTk.PhotoImage(image)
        # 显示图像
        image_label.configure(image=image_tk)
        image_label.image = image_tk

    # 创建按钮
    generate_button = tk.Button(root, text='生成图像', command=generate_image)
    generate_button.pack()

    # 创建图像显示区域
    image_frame = tk.Frame(root)
    image_frame.pack()

    image_label = tk.Label(image_frame)
    image_label.pack()

    # 运行 Tkinter 界面
    root.mainloop()



if __name__ == '__main__':
    # 数据处理
    train_set = MNIST('./mnist', train=True, download=True, transform=preprocess_img)
    test_set = MNIST('./mnist', train=False, download=True, transform=preprocess_img)
    dataset = train_set + test_set
    print(len(dataset))

    train_data = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # 定义网路
    D = Discriminator(num_classes=10).cuda()
    G = Generator(num_classes=10).cuda()
    D.apply(weights_init)
    G.apply(weights_init)

    D_optim = get_optimizer(D)
    G_optim = get_optimizer(G)
    # 标签
    label_onehots, label_fills = create_one_hot_labels(num_classes=10, image_size=32)
    # 画图，原图，和对比图
    # draw_real_images(dataset, label_onehots)
    # 训练
    train_a_gan(train_data, D, G, D_optim, G_optim, discriminator_loss, generator_loss, label_onehots, label_fills)
    # 前端
    # run_gui(label_onehots)
