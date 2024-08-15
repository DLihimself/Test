import copy
import torch
import torch.nn as nn
from torch.nn import functional as F


class Predictor(nn.Module):
    """
    用CNN 1D构建检测模块的预测分类器
    """

    def __init__(self, out_chs=16, expand=4):
        super(Predictor, self).__init__()
        self.model = nn.Sequential(
            #Input size is (1, 5)

            # Convolutional layer-1
            nn.Conv1d(in_channels=1, out_channels=out_chs, kernel_size=2, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(out_chs*expand, 2),
            nn.ReLU(),
        )

    def forward(self, indata):
        return self.model(indata)

class PredictorOriginal(nn.Module):
    def __init__(self, out_chs=16, expand=4):
        super(PredictorOriginal, self).__init__()
        self.model = nn.Sequential(
            #Input size is (1, 5)

            # Convolutional layer-1
            nn.Conv1d(in_channels=1, out_channels=out_chs, kernel_size=3, padding=1, stride=1),
            nn.MaxPool1d(2),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(out_chs*expand, 2),
        )

    def forward(self, indata):
        return self.model(indata)

class PredictorWithDropout(nn.Module):
    def __init__(self, out_chs=16, expand=4):
        super(PredictorWithDropout, self).__init__()
        self.model = nn.Sequential(
            #Input size is (1, 5)

            # Convolutional layer-1
            nn.Conv1d(in_channels=1, out_channels=out_chs, kernel_size=3, padding=1, stride=1),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(out_chs*expand, 2),
        )

    def forward(self, indata):
        return self.model(indata)

class PredictorWithReluDropout(nn.Module):
    def __init__(self, out_chs=16, expand=4):
        super(PredictorWithReluDropout, self).__init__()
        self.model = nn.Sequential(
            #Input size is (1, 5)

            # Convolutional layer-1
            nn.Conv1d(in_channels=1, out_channels=out_chs, kernel_size=2, padding=0, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            
            # Classifier
            nn.Flatten(),
            nn.Linear(out_chs*expand, 2),
            nn.ReLU(),
        )

    def forward(self, indata):
        return self.model(indata)

class Encoder(nn.Module):
    """
    用CNN 1D构建VAE的编码器
    """

    def __init__(self, in_channels, latent_dim, hidden_dims):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.hidden_dims = hidden_dims
        self.encoder_block = self.build_encoder()
        self.fc_mu = nn.Linear(hidden_dims[-1]*2, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*2, latent_dim)

    def build_encoder(self):
        modules = []
        in_channels = self.in_channels

        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        modules.append(nn.Flatten(start_dim=1))

        return nn.Sequential(*modules)

    def forward(self, indata):
        # 对输入数据进行编码
        result = self.encoder_block(indata)

        # 将数据拆分成均值和方差的形式
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]


class Decoder(nn.Module):
    """
    用CNN 1D构建VAE的解码器
    """

    def __init__(self, out_channels, latent_dim, hidden_dims):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.hidden_dims = copy.deepcopy(hidden_dims)
        self.hidden_dims.reverse()

        self.decoder_input = nn.Linear(self.latent_dim, 
                                       self.hidden_dims[0] * 2)
        self.decoder_block = self.build_decoder()
        self.final_layer = self.build_final_layer()

    def build_decoder(self):
        modules = []

        for i in range(len(self.hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(self.hidden_dims[i],
                                       self.hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm1d(self.hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        return nn.Sequential(*modules)

    def build_final_layer(self):
        last_hidden_dim = self.hidden_dims[-1]
        return nn.Sequential(
                nn.ConvTranspose1d(last_hidden_dim,
                               last_hidden_dim,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
                nn.BatchNorm1d(last_hidden_dim),
                nn.LeakyReLU(),
                nn.Conv1d(last_hidden_dim, 
                          out_channels= self.out_channels,
                          kernel_size= 6, 
                          padding= 1),
                nn.Tanh())

    def forward(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 2)
        result = self.decoder_block(result)
        result = self.final_layer(result)

        return result



class BetaVAE(nn.Module):

    num_iter = 0 # 全局变量用于追踪迭代次数

    def __init__(self,
                 in_channels: int = 1,
                 latent_dim: int = 8,
                 hidden_dims: list = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'H',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [16, 32]

        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        self.encoder = Encoder(self.in_channels, 
                               self.latent_dim, 
                               self.hidden_dims)
        self.decoder = Decoder(self.in_channels,
                               self.latent_dim,
                               self.hidden_dims)

    def save_encoder_decoder(self, save_path):
        torch.save(self.encoder.state_dict(), save_path + 'encoder.pth')
        torch.save(self.decoder.state_dict(), save_path + 'decoder.pth')

    def load_encoder_decoder(self, load_path):
        self.encoder.load_state_dict(torch.load(load_path + 'encoder.pth'))
        self.decoder.load_state_dict(torch.load(load_path + 'decoder.pth'))

    def encode(self, indata):
        """
        将输入传进encoder进行编码
        """
        return self.encoder(indata)
        

    def decode(self, z):
        """
        将laten变量z传入decoder进行解码
        """
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, indata):
        mu, log_var = self.encode(indata)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), indata, mu, log_var]

    def loss_function(self,
                      recons, 
                      indata, 
                      mu, 
                      log_var, 
                      kld_weight) -> dict:
        self.num_iter += 1
        recons_loss =F.mse_loss(recons, indata)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(indata.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x):
        return self.forward(x)[0]

    def reconstruct(self, data_corrupt, sample_type='sample'):
        z_mu, z_log_var = self.encode(data_corrupt)
        
        if sample_type == 'sample':
            z = self.reparameterize(z_mu, z_log_var)
            data_impute = self.decode(z)
        elif sample_type == 'mean':
            data_impute = self.decoder(z_mu)
        else:
            raise ValueError('Undefined sample type.')

        return data_impute
