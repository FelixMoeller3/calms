from torch import nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
from torch import autograd

EPSILON = 1e-16


class GANCritic(nn.Module):
    def __init__(self, image_size:int, num_channels: int):
        '''
            :param image_size: The size of a single side of the image (assumed to be square)
            :param num_channels: The number of channels in the image
        '''
        # configurations
        super().__init__()
        self.image_size = image_size
        self.num_channels = num_channels
        self.channel_size = 64

        # layers
        self.conv1 = nn.Conv2d(
            num_channels, self.channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            self.channel_size, self.channel_size*2,
            kernel_size=4, stride=2, padding=1
        )
        self.conv3 = nn.Conv2d(
            self.channel_size*2, self.channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.conv4 = nn.Conv2d(
            self.channel_size*4, self.channel_size*8,
            kernel_size=4, stride=1, padding=1,
        )
        self.fc = nn.Linear((image_size//8)**2 * self.channel_size*4, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.view(-1, (self.image_size//8)**2 * self.channel_size*4)
        return self.fc(x)


class GANGenerator(nn.Module):
    def __init__(self, z_size, image_size, num_channels):
        '''
            :param z_size: The size of the latent vector
            :param image_size: The size of a single side of the image (assumed to be square)
            :param num_channels: The number of channels in the image
        '''
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.channel_size = 64

        # layers
        self.fc = nn.Linear(z_size, (image_size//8)**2 * self.channel_size*8)
        self.bn0 = nn.BatchNorm2d(self.channel_size*8)
        self.bn1 = nn.BatchNorm2d(self.channel_size*4)
        self.deconv1 = nn.ConvTranspose2d(
            self.channel_size*8, self.channel_size*4,
            kernel_size=4, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm2d(self.channel_size*2)
        self.deconv2 = nn.ConvTranspose2d(
            self.channel_size*4, self.channel_size*2,
            kernel_size=4, stride=2, padding=1,
        )
        self.bn3 = nn.BatchNorm2d(self.channel_size)
        self.deconv3 = nn.ConvTranspose2d(
            self.channel_size*2, self.channel_size,
            kernel_size=4, stride=2, padding=1
        )
        self.deconv4 = nn.ConvTranspose2d(
            self.channel_size, num_channels,
            kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        g = F.relu(self.bn0(self.fc(z).view(
            z.size(0),
            self.channel_size*8,
            self.image_size//8,
            self.image_size//8,
        )))
        g = F.relu(self.bn1(self.deconv1(g)))
        g = F.relu(self.bn2(self.deconv2(g)))
        g = F.relu(self.bn3(self.deconv3(g)))
        g = self.deconv4(g)
        return torch.sigmoid(g)
    

class WGAN():
    def __init__(self,
                 image_size:int, num_channels:int,z_size:int=100,use_gpu:bool=False):
        # configurations
        super().__init__()
        self.z_size = z_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.use_gpu = use_gpu
        # components
        self.critic = GANCritic(
            image_size=self.image_size,
            num_channels=self.num_channels
        )
        if self.use_gpu:
            self.critic.cuda()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=1e-4,betas=(0.5,0.9),weight_decay=1e-5)
        self.generator = GANGenerator(
            z_size=self.z_size,
            image_size=self.image_size,
            num_channels=self.num_channels
        )
        if self.use_gpu:
            self.generator.cuda()
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),lr=1e-4,betas=(0.5,0.9),weight_decay=1e-5)
        self.lamda = 10.0

    def train_a_batch(self, train_x:torch.Tensor, generated_x:torch.Tensor, importance_of_new_task=.5):
        assert generated_x is None or train_x.size() == generated_x.size()

        # run the critic and backpropagate the errors.
        self.critic_optimizer.zero_grad()
        z = self._noise(train_x.size(0))

        # run the critic on the real data.
        c_loss_real, g_real = self._critic_loss(train_x, z, return_g=True)
        c_loss_real_gp = (
            c_loss_real + self._gradient_penalty(train_x, g_real, self.lamda)
        )

        # run the critic on the replayed data.
        if generated_x is not None:
            c_loss_replay, g_replay = self._critic_loss(generated_x, z, return_g=True)
            c_loss_replay_gp = (c_loss_replay + self._gradient_penalty(
                generated_x, g_replay, self.lamda
            ))
            c_loss = (
                importance_of_new_task * c_loss_real +
                (1-importance_of_new_task) * c_loss_replay
            )
            c_loss_gp = (
                importance_of_new_task * c_loss_real_gp +
                (1-importance_of_new_task) * c_loss_replay_gp
            )
        else:
            c_loss = c_loss_real
            c_loss_gp = c_loss_real_gp

        c_loss_gp.backward()
        self.critic_optimizer.step()

        # run the generator and backpropagate the errors.
        self.generator_optimizer.zero_grad()
        z = self._noise(train_x.size(0))
        g_loss = self._generator_loss(z)
        g_loss.backward()
        self.generator_optimizer.step()

        return {'c_loss': c_loss.item(), 'g_loss': g_loss.item()}

    def sample(self, size):
        return self.generator(self._noise(size))

    def _noise(self, size):
        z = Variable(torch.randn(size, self.z_size)) * .1
        return z.cuda() if self.use_gpu else z

    def _critic_loss(self, x, z, return_g=False):
        g = self.generator(z)
        c_x = self.critic(x).mean()
        c_g = self.critic(g).mean()
        l = -(c_x-c_g)
        return (l, g) if return_g else l

    def _generator_loss(self, z, return_g=False):
        g = self.generator(z)
        l = -self.critic(g).mean()
        return (l, g) if return_g else l

    def _gradient_penalty(self, x, g, lamda):
        assert x.size() == g.size()
        a = torch.rand(x.size(0), 1)
        a = a.cuda() if self.use_gpu else a
        a = a\
            .expand(x.size(0), x.nelement()//x.size(0))\
            .contiguous()\
            .view(
                x.size(0),
                self.num_channels,
                self.image_size,
                self.image_size
            )
        interpolated = Variable(a*x.data + (1-a)*g.data, requires_grad=True)
        c = self.critic(interpolated)
        gradients = autograd.grad(
            c, interpolated, grad_outputs=(
                torch.ones(c.size()).cuda() if self.use_gpu else
                torch.ones(c.size())
            ),
            create_graph=True,
            retain_graph=True,
        )[0]
        return lamda * ((1-(gradients+EPSILON).norm(2, dim=1))**2).mean()
