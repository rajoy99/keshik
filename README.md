# keshik


<img src="https://github.com/rajoy99/keshik/blob/main/KeshikLogo.png"/> 
Oversample class imbalanced tabular data by Denoising Diffusion Probabilistic Model (DDPM)

**Usage Guide**
---

Installation:

```python
pip install keshik
```

The APIs are very simple to use. This example shows the usage : 

```python

from keshik.oversampler import Diffusion_Oversampler

totalX,totalY=Diffusion_Oversampler(X_train,y_train,n_iteration=40,categorical_columns=['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'])


```

Method : Diffusion_Oversampler 


Input:  
* X_train : Dataframe of Feature Variables 
* y_train : Series of binary labels 
* n_iteration: number of epochs to train the DDPM model 
* categorical_columns : list of categorical columns 
* cat_handle_mode : ['prediction','roundoff'] (default : 'prediction')
* oversampling_mode : ['byMatrix','byRows','byColumns'] (default : 'byMatrix')

Returns : 
* A dataframe with original features 
* a numpy array (the binary labels) 
    

