import torch
import torch.nn as nn
from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np




def scheduler(beta1, beta2, T) :

    """
    
    Computes schedules for training and sampling. 
    Inputs:
    beta1 : float, within range(0,1)
    beta2 : float, within range(0,1)
    T : int, number of perturbation iterations
    Returns : Dict[str,torch.Tensor]

    """


    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  
        "oneover_sqrta": oneover_sqrta,  
        "sqrt_beta_t": sqrt_beta_t,  
        "alphabar_t": alphabar_t,  
        "sqrtab": sqrtab,  
        "sqrtmab": sqrtmab,  
        "mab_over_sqrtmab": mab_over_sqrtmab_inv
    }


# Template Layer for use in auxiliary model: Convolution -> BatchNorm -> LeakyRelU
CBL_layer = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)


class Auxiliary(nn.Module):

    """
        Predicts the Noise for the reverse markov chain
        Any Mathematical function (or neural network) that maps from n-dimensional space 
        to n-dimensional space

    """

    def __init__(self, n_channel: int) -> None:
        super(Auxiliary, self).__init__()
        self.conv = nn.Sequential(  
            CBL_layer(n_channel, 64),
            CBL_layer(64, 128),
            CBL_layer(128, 256),
            CBL_layer(256, 512),
            CBL_layer(512, 256),
            CBL_layer(256, 128),
            CBL_layer(128, 64),
            nn.Conv2d(64, n_channel, 3, padding=1),
        )

    def forward(self, x, t) -> torch.Tensor:
        

        return self.conv(x)




class Diffusion(nn.Module):

    def __init__(
        self,
        Epsilon_Predictor: nn.Module,
        betas: Tuple[float, float],
        n_T: int,
        criterion: nn.Module = nn.MSELoss(),
    ) -> None:
        super(Diffusion, self).__init__()
        self.Epsilon_Predictor = Epsilon_Predictor


        for k, v in scheduler(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = criterion


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """

        Training Process
        1. Implements forward markov chain of Diffusion
        2. Predicts noise value from x_t using Epsilon_Predictor
        3. Returns the loss of the Epsilon Predictor(or the Auxiliary Model)
        
        """


        # Sample t from Uniform Distribution : u(0, n_T)
        _ts = torch.randint(1, self.n_T, (x.shape[0],)).to(
            x.device
        )  


        # Sample epsilon from Normal Distribution : N(0, 1)
        eps = torch.randn_like(x)  


        # Compute x_t from Diffusion's forward chain formula
        x_t = (
            self.sqrtab[_ts, None, None, None] * x + self.sqrtmab[_ts, None, None, None] * eps
        )  
        
        # Predict the error term from the auxiliary model and return the loss
        return self.criterion(eps, self.Epsilon_Predictor(x_t, _ts / self.n_T))




    def sample(self, desired_samples: int, size, device):

        """

        Generates new data by the reverse Diffusion process
        Input : 
        desired_samples : Type (int)
        size : Type (tuple)

        """

        # Sample x_T from Normal Distribution : N(0, 1)
        x_i = torch.randn(desired_samples, *size).to(device)  


        # Sampling Algorithm in DDPM Paper. 
        for i in range(self.n_T, 0, -1):

            z = torch.randn(desired_samples, *size).to(device) if i > 1 else 0
            eps = self.Epsilon_Predictor(x_i, i / self.n_T)

            #Reverse Diffusion Process. Will run Until i=1. x_1 is the Generated Sample
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])   + self.sqrt_beta_t[i] * z
            )

        return x_i
