import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as Data
from torchsummary import summary
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from modules.models import BetaVAE

class RepairModule():
    """
    修复模块，填充缺失数据
    """

    def __init__(self, 
                 n_epoch = 100, 
                 kld_weight=0.00025):
        self.n_epoch = n_epoch
        self.kld_weight = kld_weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae = BetaVAE().to(self.device)

    def get_data_loader(self, data, batch_size):
        datasets = np.array(data)
        data_train = Data.TensorDataset(
            torch.tensor(datasets, dtype=torch.float32))
        train_loader = Data.DataLoader(
            dataset=data_train,
            batch_size=batch_size,
            shuffle=True)
        return train_loader

    def train_one_epoch(self, optimizer, train_loader, i_epoch, epochs):
        total_samples = 0.0
        loss_sum = 0.0

        self.vae.train()
        tbar = tqdm(train_loader, desc='Train')
        for titem in tbar:
            dt = titem[0]
            dt.resize_(dt.size()[0], 1, dt.size()[1])
            dt = dt.to(self.device)

            # forward
            results = self.vae(dt) # results is [recons, indata, mu, log_var]
            loss_dict = self.vae.loss_function(*results, self.kld_weight)
            loss = loss_dict['loss']

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gardient descent or adam step
            optimizer.step()

            # calculate train loss
            with torch.no_grad():
                total_samples += dt.size(0)
                loss_sum += loss.item()
                running_loss = loss_sum / total_samples

            # update tbar info
            tbar.set_description(f'Train Epoch [{i_epoch+1}/{epochs}]')
            tbar.set_postfix({'loss': f'{running_loss*1e4:.4f}e-4'})

        train_epoch_loss = loss_sum / total_samples
        return train_epoch_loss

    def train(self, data, batch_size=16, epochs=50, lr=0.005):
        train_loader = self.get_data_loader(data, batch_size)
        optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr)

        _, n_features = train_loader.dataset.tensors[0].shape
        summary(self.vae, input_size=(1, n_features))

        train_results = []
        for i_epoch in range(epochs):
            train_loss = self.train_one_epoch(optimizer, train_loader, i_epoch, epochs)
            train_results.append(train_loss)

        train_results = pd.DataFrame(train_results, 
                                     columns=['loss'], 
                                     index=pd.Index(range(1, epochs+1), name='epochs'))
        return train_results

    def impute(self, data_corrupt, iters=10):
        indata = data_corrupt
        for i in range(iters):
            data_impute = self.vae.reconstruct(indata)
            indata = data_impute
        return data_impute

    def save_model(self, save_path):
        self.vae.save_encoder_decoder(save_path)

    def load_model(self, load_path):
        encoder_exists = os.path.exists(load_path + 'encoder.pth')
        decoder_exists = os.path.exists(load_path + 'decoder.pth')
        if encoder_exists and decoder_exists:
            self.vae.load_encoder_decoder(load_path)
