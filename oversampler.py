import numpy as np
import numpy as np
from sklearn.preprocessing import MinMaxScaler,RobustScaler
from traindiffusion_byMatrix import train_Diffusion_byMatrix
from traindiffusion_byRows import train_Diffusion_byRows
from traindiffusion_byColumns import train_Diffusion_byColumns
import pandas as pd 
from sklearn.neural_network import MLPClassifier


def mapper(t,a,b,c,d):

    '''
    Maps a value to Desired range
    '''
    return c+((d-c)*(t-a)/(b-a))




def Diffusion_Oversampler(X_train,y_train,n_iteration=2,categorical_columns=[],n_T=1000,oversampling_mode='byMatrix',cat_handle_mode='prediction'):


    cols=X_train.shape[1]
    col_names=X_train.columns

    #minority determination
    minority_class=y_train.value_counts().index[-1]
    majority_class=y_train.value_counts().index[0]
    print("Minority and Majority class are: ",minority_class,majority_class)
    desired=y_train.value_counts().iloc[0] - y_train.value_counts().iloc[1]

    original_data_minority=X_train[y_train==minority_class]
    original_data_majority=X_train[y_train==majority_class]

    #Scaling
    scaler=RobustScaler()
    scaled_minority=scaler.fit_transform(original_data_minority)




    # Synthetic Data Generation
    if oversampling_mode=='byMatrix':
        generated=train_Diffusion_byMatrix(data=scaled_minority,n_iteration=n_iteration,augment_size=desired,n_T=n_T)
    elif oversampling_mode=='byColumns':
        generated=train_Diffusion_byColumns(data=scaled_minority,n_iteration=n_iteration,augment_size=desired,n_T=n_T)
    elif oversampling_mode=='byRows':
        generated=train_Diffusion_byRows(data=scaled_minority,n_iteration=n_iteration,augment_size=desired,n_T=n_T)


    #Inverse Scaling
    generated_rescaled=scaler.inverse_transform(generated)
    generatedDF = pd.DataFrame(generated_rescaled, columns=col_names) 
    
    
    # Handling Categorical variables 
    if cat_handle_mode=='prediction':
        numerical_columns=list(set(list(X_train.columns))-set(categorical_columns))
        if len(categorical_columns)>0:
            Classifier=MLPClassifier()
            TrainX=original_data_minority[numerical_columns]
            TestX=generatedDF[numerical_columns]

            for v in categorical_columns:

                TrainY=original_data_minority[v]
                Classifier.fit(TrainX,TrainY)

                generatedDF[v]=Classifier.predict(TestX)

    elif cat_handle_mode=='roundoff':
        if len(categorical_columns)>0:
            for v in categorical_columns:
                generatedDF[v]=generatedDF[v].map(np.round)


    frames = [generatedDF, original_data_minority, original_data_majority]
    totalX=pd.concat(frames)
    totalY=np.concatenate((np.full(generated_rescaled.shape[0]+original_data_minority.shape[0], minority_class),np.full(original_data_majority.shape[0],majority_class)),axis=None)


    return totalX,totalY
