#coding: utf-8
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from modules.models import Predictor


class DetectModule():
    def __init__(self):
        print('init detect module')
        self.train_loader = None 
        self.test_loader = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Predictor(expand=2).to(self.device)

    def get_data_loader(self, data, batch_size):
        datasets = np.array(data.iloc[:, 0:-1])
        labels = np.array(data['is_fault']).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(labels)
        labels = enc.transform(labels).toarray()
        x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2,random_state=0)
        data_train = Data.TensorDataset(
            torch.tensor(x_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.float32))
        train_loader = Data.DataLoader(
            dataset=data_train,
            batch_size=batch_size,
            shuffle=True)
        data_test = Data.TensorDataset(
            torch.tensor(x_test, dtype=torch.float32), 
            torch.tensor(y_test, dtype=torch.float32))
        test_loader = Data.DataLoader(
            dataset=data_test,
            batch_size=batch_size,
            shuffle=True)

        return train_loader, test_loader

    def train_one_epoch(self, optimizer, criterion, i_epoch, epochs):
        # ========Train Mode========
        correct_samples = 0.0
        total_samples = 0.0
        loss_sum = 0.0

        self.model.train()
        tbar = tqdm(self.train_loader, desc='Train')
        for dt, labels in tbar:
            # Resize data and get data to CUDA if available
            dt.resize_(dt.size()[0], 1, dt.size()[1])
            dt, labels = dt.to(self.device), labels.to(self.device)

            # forward
            labels_pred = self.model(dt)
            loss = criterion(labels_pred, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gardient descent or adam step
            optimizer.step()

            # calculate train loss and train acc
            with torch.no_grad():
                labels_p_idx = torch.argmax(labels_pred, dim=1) # predicted labels
                labels_a_idx = torch.argmax(labels, dim=1) # actual labels 
                correct_samples += (labels_p_idx == labels_a_idx).sum().item()
                total_samples += labels.size(0)
                loss_sum += loss.item()

                running_loss = loss_sum / total_samples
                running_acc = correct_samples / total_samples

            # update tbar info
            tbar.set_description(f'Train Epoch [{i_epoch+1}/{epochs}]')
            tbar.set_postfix({'loss': f'{running_loss:.4f}', 'acc': f'{running_acc*100:.2f}%'})

        train_epoch_loss = loss_sum / total_samples
        train_epoch_acc = correct_samples / total_samples

        return train_epoch_loss, train_epoch_acc


    def test_one_epoch(self, criterion):
        # ========Test Mode========
        correct_samples = 0.0
        total_samples = 0.0
        loss_sum = 0.0

        self.model.eval()
        tbar = tqdm(self.test_loader, desc='Test')
        for dt, labels in tbar:
            # Resize data and get data to CUDA if available
            dt.resize_(dt.size()[0], 1, dt.size()[1])
            dt, labels = dt.to(self.device), labels.to(self.device)
            with torch.no_grad():
                labels_pred = self.model(dt)
                loss = criterion(labels_pred, labels)
                labels_p_idx = torch.argmax(labels_pred, dim=1) # predicted labels
                labels_a_idx = torch.argmax(labels, dim=1) # actual labels 
                correct_samples += (labels_p_idx == labels_a_idx).sum().item()
                total_samples += labels.size(0)
                loss_sum += loss.item()

                running_loss = loss_sum / total_samples
                running_acc = correct_samples / total_samples

            # update tbar info
            tbar.set_postfix({'loss': f'{running_loss:.4f}', 'acc': f'{running_acc*100:.2f}%'})

        test_epoch_loss = loss_sum / total_samples
        test_epoch_acc = correct_samples / total_samples

        return test_epoch_loss, test_epoch_acc

    def train(self, data, batch_size=16, epochs=50, lr=0.001):
        self.train_loader, self.test_loader = self.get_data_loader(data, batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        _, n_features = self.train_loader.dataset.tensors[0].shape
        summary(self.model, input_size=(1, n_features))

        train_results = []
        test_results = []
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(optimizer, criterion, epoch, epochs)
            test_loss, test_acc = self.test_one_epoch(criterion)
            train_results.append((train_loss, train_acc))
            test_results.append((test_loss, test_acc))

        train_results = pd.DataFrame(train_results, 
                                     columns=['loss', 'acc'], 
                                     index=pd.Index(range(1, epochs+1), name='epochs'))
        test_results = pd.DataFrame(test_results, 
                                    columns=['loss', 'acc'], 
                                    index=pd.Index(range(1, epochs+1), name='epochs'))

        return train_results, test_results


    def predict(self, data):
        labels_pred = self.model(data)
        labels = torch.argmax(labels_pred, dim=1)
        return labels.numpy()

    def save_model(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load_model(self, load_path):
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path))


    
