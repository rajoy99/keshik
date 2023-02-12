import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np
from DDPM_model import Diffusion,Auxiliary


def train_Diffusion_byMatrix(data,n_iteration,augment_size,n_T=1000):

    '''
    takes One dataframe (Features )and one series(target column 0/1) as input     
    Returns generated data in desired 
    '''

    #square matrix generation 
    squared=[]
    rows=data.shape[0]
    cols=data.shape[1]

    i=0
    while i+cols<rows:
        squared.append(data[i:i+cols])
        i+=cols

    data=np.array(squared) 
    
    cols=data.shape[1]
    data=torch.Tensor(data)
    un_dataloader=DataLoader(data)


    n_epoch = n_iteration
    device="cuda:0"
    ddpm = Diffusion(Epsilon_Predictor=Auxiliary(1), betas=(1e-4, 0.02), n_T=n_T)
    ddpm.to(device)

    generated_size=0

    optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

    for i in range(n_epoch):

        ddpm.train()
        
        progress_bar = tqdm(un_dataloader)
  
        loss_ema = None

        for x in progress_bar:
            optim.zero_grad()
            # print("Shape of the fed tensor : ",x.shape)
            x = x.to(device)
            loss = ddpm(x)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            progress_bar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()
            

        ddpm.eval()


    generated=[]
    while generated_size<augment_size:
        
        with torch.no_grad():
            xh = ddpm.sample(1, (1,cols,cols), device)
            xh=xh.cpu().detach().numpy().reshape(cols,cols)
            # xh=scaler.inverse_transform(xh)

            generated.append(xh)
            
            print("TG len",len(generated))
            generated_size+=cols
    
    generated=np.vstack(generated)
    print("Generated Shape : ",generated.shape)

    return generated