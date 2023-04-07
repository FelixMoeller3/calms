from torch import nn
from torch.nn import functional as F
import torch
from torch.autograd import Variable
from torch import autograd
from tqdm import tqdm
from torch.utils.data import Dataset,DataLoader

EPSILON = 1e-16


class GANCritic(nn.Module):
    def __init__(self, image_size:int, num_channels: int):
        '''
            :param image_size: The size of a single side of the image (assumed to be square)
            :param num_channels: The number of channels in the image
        '''
        # configurations
        super().__init__()
        assert image_size >=32, "Image size must be at least 32"
        self.image_size = image_size
        self.num_channels = num_channels
        self.channel_size = 64

        # layers
        self.main_module = nn.Sequential(
            nn.Conv2d(num_channels, self.channel_size*4,kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel_size*4,affine=True),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(self.channel_size*4, self.channel_size*8,kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel_size*8,affine=True),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(self.channel_size*8, self.channel_size*16,kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.channel_size*16,affine=True),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(self.channel_size*16, 1,kernel_size=4, stride=1, padding=0),
        )
        self.fc = nn.Linear((image_size//32)**2, 1)

    def forward(self, x):
        x = self.main_module(x)
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


        self.main_module = nn.Sequential(
            nn.ConvTranspose2d(self.z_size, self.channel_size*16,kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(self.channel_size*16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.channel_size*16, self.channel_size*8,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_size*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.channel_size*8, self.channel_size*4,kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(self.channel_size*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.channel_size*4, self.num_channels,kernel_size=4, stride=2, padding=1),
        )
        self.output = nn.Tanh()

    def forward(self, z):
        z = z.view(z.size(0),z.size(1),1,1)
        z = self.main_module(z)
        return self.output(z)    

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
        self.critic_iter = 5
        self.gen_iter = 100
        learning_rate = 2e-4
        beta1 = 0.0
        beta2 = 0.9
        if self.use_gpu:
            self.critic.cuda()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=learning_rate,betas=(beta1,beta2))
        self.generator = GANGenerator(
            z_size=self.z_size,
            image_size=self.image_size,
            num_channels=self.num_channels
        )
        if self.use_gpu:
            self.generator.cuda()
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(),lr=learning_rate,betas=(beta1,beta2))
        self.lamda = 10.0

    def train(self, num_epochs:int,prev_generator, train_loader: DataLoader, importance_of_new_task=.2):

        for i in range(num_epochs):
            print(f"Running generator epoch {i+1}/{num_epochs}")
            generator_loss = 0.0
            discriminator_loss = 0.0
            train_data = self._get_infinite_batches(train_loader)
            for gen_iter in tqdm(range(self.gen_iter),desc="Training WGAN"):
                # Train discriminator
                discriminator_loss += self._train_critic(train_data,prev_generator,importance_of_new_task)
                # Train generator
                self.generator_optimizer.zero_grad()
                z = self._noise(next(train_data).size(0))
                g_loss = self._generator_loss(z)
                g_loss.backward()
                self.generator_optimizer.step()
                generator_loss += g_loss.item()



            generator_loss /= gen_iter
            discriminator_loss /= gen_iter
            if prev_generator is not None:
                discriminator_loss /= 2
            print('Generator Loss: {:.4f} Discriminator Loss: {:.4f}'.format(generator_loss, discriminator_loss))

    
    def _train_critic(self,train_data,prev_generator,importance_of_new_task:float=0.2) -> float:
        critic_loss = 0.0
        for _ in range(self.critic_iter):
            data = next(train_data)
            z = self._noise(data.size(0))
            if self.use_gpu:
                data = data.cuda()
            self.critic_optimizer.zero_grad()
            c_loss_real, g_real = self._critic_loss(data, z, return_g=True)
            c_loss_real_gp = (
                c_loss_real + self._gradient_penalty(data, g_real, self.lamda)
            )

            # run the critic on the replayed data.
            if prev_generator is not None:
                gen_data = prev_generator.sample(data.size(0))
                c_loss_replay, g_replay = self._critic_loss(gen_data, z, return_g=True)
                c_loss_replay_gp = (c_loss_replay + self._gradient_penalty(
                    gen_data, g_replay, self.lamda
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
            critic_loss += c_loss.item()

        return critic_loss/self.critic_iter
    
    def _get_infinite_batches(self, data_loader):
        while True:
            for data,_ in data_loader:
                yield data

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
