# keshik
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

