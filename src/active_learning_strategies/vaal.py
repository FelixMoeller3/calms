from .strategy import Strategy
import random
import numpy as np
import torch.nn as nn
import torch
from torch.optim import Adam
from torch.utils.data import Dataset,DataLoader
from models.vaal_models import VAE,Discriminator
from data import SubsetSequentialSampler
from tqdm import tqdm

class VAAL(Strategy):

    def __init__(self, model: nn.Module, data_unlabeled:Dataset, NO_CLASSES: int,BATCH:int,BUDGET:int,INIT_BUDGET:int,
                  LOOKBACK:int, USE_GPU:bool=False, TRAIN_EPOCHS:int=100, BETA:float=1.0, ADVERSARY_PARAM:float=1.0,**kwargs):
        super(VAAL, self).__init__(model, data_unlabeled, NO_CLASSES,BATCH,BUDGET,INIT_BUDGET,LOOKBACK,USE_GPU)
        input_shape = data_unlabeled[0][0].shape
        self.vae = VAE(input_shape[1],input_shape[0])
        self.discriminator = Discriminator(input_shape[1])
        self.num_epochs = TRAIN_EPOCHS
        self.beta = BETA
        self.adversary_param = ADVERSARY_PARAM
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        if self.use_gpu:
            self.vae = self.vae.cuda()
            self.discriminator = self.discriminator.cuda()

    def query(self) -> np.ndarray:
        if len(self.subset) <= self.BUDGET:
            arg = np.array([i for i in range(len(self.subset))])
        else:
            self._train_step()
            arg = self._query_step()
        self.add_query(arg[:self.BUDGET])
        return np.concatenate(self.previous_queries)
    
    def read_data(self, dataloader):
        while True:
            for img, label in dataloader:
                yield img, label

    def _train_step(self) -> None:
        print("Training VAE and discriminator...")
        self.optim_vae = Adam(self.vae.parameters(),lr=5e-4)
        self.optim_discriminator = Adam(self.vae.parameters(), lr=5e-4)
        for epoch in range(self.num_epochs):
            print(f"Running epoch {epoch+1}/{self.num_epochs}")
            random.shuffle(self.labeled_set)
            random.shuffle(self.subset)
            labeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                        sampler=SubsetSequentialSampler(self.labeled_set), 
                                        pin_memory=True)
            unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                        sampler=SubsetSequentialSampler(self.subset), 
                                        pin_memory=True)
            labeled_gen = self.read_data(labeled_loader)
            unlabeled_gen = self.read_data(unlabeled_loader)
            for _ in tqdm(range(len(self.data_unlabeled)//self.BATCH), desc="Running iterations"):
                labeled_data = next(labeled_gen)
                unlabeled_data = next(unlabeled_gen)
                if self.use_gpu:
                    labeled_data = labeled_data[0].cuda(), labeled_data[1].cuda()
                    unlabeled_data = unlabeled_data[0].cuda(), unlabeled_data[1].cuda()
                self._train_vae(labeled_data,unlabeled_data)
                self._train_discriminator(labeled_data,unlabeled_data)
        print("Finished training VAE and discriminator")
    

    def _train_vae(self,labeled_data: tuple[torch.Tensor], unlabeled_data: tuple[torch.Tensor]) -> None:
        labeled_imgs, _ = labeled_data
        unlabeled_imgs, _ = unlabeled_data
        recon, _, mu, logvar = self.vae(labeled_imgs)
        unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.beta)
        unlab_recon, _, unlab_mu, unlab_logvar = self.vae(unlabeled_imgs)
        transductive_loss = self.vae_loss(unlabeled_imgs, 
                unlab_recon, unlab_mu, unlab_logvar, self.beta)
    
        labeled_preds = self.discriminator(mu)
        unlabeled_preds = self.discriminator(unlab_mu)
        
        lab_real_preds = torch.ones_like(labeled_preds)
        unlab_real_preds = torch.zeros_like(unlabeled_preds)
            
        if self.use_gpu:
            lab_real_preds = lab_real_preds.cuda()
            unlab_real_preds = unlab_real_preds.cuda()

        dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                self.bce_loss(unlabeled_preds, unlab_real_preds)
        total_vae_loss = unsup_loss + transductive_loss + self.adversary_param * dsc_loss
        self.optim_vae.zero_grad()
        total_vae_loss.backward()
        self.optim_vae.step()

    def _train_discriminator(self, labeled_data: tuple[torch.Tensor], unlabeled_data: tuple[torch.Tensor]) -> None:
        labeled_imgs, _ = labeled_data
        unlabeled_imgs, _ = unlabeled_data
        with torch.no_grad():
            _, _, mu, _ = self.vae(labeled_imgs)
            _, _, unlab_mu, _ = self.vae(unlabeled_imgs)
        
        labeled_preds = self.discriminator(mu)
        unlabeled_preds = self.discriminator(unlab_mu)
        
        lab_real_preds = torch.ones_like(labeled_preds)
        unlab_fake_preds = torch.zeros_like(unlabeled_preds)

        if self.use_gpu:
            lab_real_preds = lab_real_preds.cuda()
            unlab_fake_preds = unlab_fake_preds.cuda()
        
        dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                self.bce_loss(unlabeled_preds, unlab_fake_preds)

        self.optim_discriminator.zero_grad()
        dsc_loss.backward()
        self.optim_discriminator.step()

    def _query_step(self) -> None:
        unlabeled_loader = DataLoader(self.data_unlabeled, batch_size=self.BATCH, 
                                        sampler=SubsetSequentialSampler(self.subset), 
                                        pin_memory=True)
        all_preds = []

        for images, _ in unlabeled_loader:
            if self.use_gpu:
                images = images.cuda()

            with torch.no_grad():
                _, _, mu, _ = self.vae(images)
                preds = self.discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator thinks are the most likely to be unlabeled
        return np.argsort(all_preds)
    

    def vae_loss(self, x: torch.Tensor, recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD