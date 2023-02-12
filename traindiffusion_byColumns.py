import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from tqdm import tqdm
import numpy as np


from DDPM_model import Diffusion,Auxiliary
from sklearn.preprocessing import MinMaxScaler


def train_Diffusion_byColumns(data,n_iteration,augment_size,n_T=1000):

    '''

    takes One dataframe (Features )and one series(target column 0/1) as input     
    Returns generated data in desired 

    '''


    rows=data.shape[0]
    cols=data.shape[1]
    data=data.T
    dataTensor=torch.Tensor(data)
    un_dataloader=DataLoader(dataTensor)

    generated=[]

    n_epoch = 1

    device="cuda:0"

    count=0

    for k in range(cols):

      scaler=MinMaxScaler((-0.99,0.99))
      currdata=data[k]
      # currdata=scaler.fit_transform(currdata.reshape(-1,1)).reshape(1,-1)
      currdataTensor=torch.Tensor(data)
      currdataLoader=DataLoader(currdataTensor)

      ddpm = Diffusion(Epsilon_Predictor=Auxiliary(1), betas=(1e-4, 0.02), n_T=n_T)
      ddpm.to(device)
      optim = torch.optim.Adam(ddpm.parameters(), lr=2e-4)

      generated_size=0


      for i in range(n_epoch):

          ddpm.train()
          
          progress_bar = tqdm(currdataLoader)
    
          loss_ema = None
          
          for x in progress_bar:
              if count==0:
                print(type(x),x.shape)
                count+=1
              optim.zero_grad()
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

      tillgenerated=[]
      while generated_size<augment_size:
        
        with torch.no_grad():
          xh = ddpm.sample(1, (1,1,rows), device)
          xh=xh.cpu().detach().numpy().reshape(-1,1)


        tillgenerated.append(xh.reshape(rows,1))
        

        generated_size+=rows
      
      tillgenerated=np.vstack(tillgenerated)
      tillgenerated=tillgenerated[:augment_size,:]

      generated.append(tillgenerated)



    generated=np.hstack(generated)
    

    print("Generated Size",generated.shape)
    return generated