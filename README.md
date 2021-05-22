# OSIC-pulmonary-fibrosis-Competetion rank - 67 ( silver medal ) My score = -6.847 , First position score = -6.8305
  https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression
  download the data from this link as the data is huge :-
  https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression/data

# What is pulmonary fibrosis ? 

Pulmonary fibrosis is a lung disease that occurs when lung tissue becomes damaged and scarred. This thickened, stiff tissue makes it more difficult for your lungs to work properly. As pulmonary fibrosis worsens, you become progressively more short of breath.

<img src="https://www.pulmonaryfibrosis.org/images/default-source/default-album/normal-and-impaired-gas-exchange.png?sfvrsn=c3b0918d_0" width=600>

# What we need to predict ?

We need to predict a patient’s severity of decline in lung function based on a CT scan of their lungs. Lung function is assessed based on output from a spirometer, which measures the forced vital capacity (FVC), i.e. the volume of air exhaled. The challenge is to use machine learning techniques to make a prediction with the image, metadata, and baseline FVC as input.


<img src="https://i.imgur.com/8AWVnqQ.png" width=650>

# Evaluation metrics :-

- The evaluation metric of this competition is a modified version of Laplace Log Likelihood. 
Predictions are evaluated with a modified version of the Laplace Log Likelihood. For each sample in test set, an `FVC` and a `Confidence` measure (standard deviation σ) has to be predicted.

![](https://i.imgur.com/tEIZvli.png)

SIGMAclipped = tf.maximum( SIGMA , 70)

DELTA = tf.minimum( tf.math.abs(FVCtrue - FVCpred),1000)

metric = (-tf.math.sqrt(2) * DELTA / SIGMAclipped) - tf.math.log(tf.math.sqrt(2)*SIGMAclipped)

# My approach :- 
My final solution is a blend / weighted ensemble of 4 models :-

A. efficientnet with b5 layer which were trained on  windowed lung ct scan images along with the meta data for 5 folds -> pool -> flatten -> dropout -> concat. And the meta model was a simple head with features -> linear -> relu -> linear -> relu -> concat. The final models either had 512->1024 or 100->100 features for the head. And finally a simple linear layer for the 3 FVC output.
the approach on this model was to use the images to predict the betas (slopes for the FVC declines ) and apply the linear decay method 
training parameters of efficient net :-

1. Adam optimizer with Reduce on Plateau scheduler
image size 520x520
2. Trained with an LR of .003 for 90 epochs
3. CV performance evaluated based on competition metrics
4. Batch size of 16 (bs of 4 for one model)
5. Saved checkpoint based on best validation score . 

B. Qunatile regressor with custom pinball loss function ( only tabular data used )
I did in depth research on pinball loss function and finally used a novel assymetric pinball loss function with epsilon value of 0.8 
because the normal pinball loss function was not giving good results as the data was very noisy 

def new_asy_qloss(y_true,y_pred):
    qs = [0.2, 0.50, 0.8]
    q = tf.constant(np.array([qs]), dtype=tf.float32)
    e = y_true - y_pred
    epsilon = 0.8

    v = tf.maximum( -(1-q)*(e+q*epsilon), q*(e-(1-q)*epsilon))
    v1 = tf.maximum(v,0.0)
    return K.mean(v1)

parametrs used while training :

1. used group kfold stratergy for 5 folds 
2. used 0.65 percent of custom pinball loss function + 0.35 of metric loss 
3. trained for 855 epochs 
4. batch size = 128 
5. Didn't use any batch accumulation or mixed precision

C. Elastic net model :-
elsatic net and lasso performed better as compared to ridge / lgbm / ngboost / bayesian ridge / huber / catboost models i tried all of them and finally used elastic net for
the final model and finally by hypertuning the parametrs with groupkfold for 10 folds for cv evaluation i did weighted ensembling 
tabular data used :- 
1. baseline_FVC
2.baseline_Percent 
3.Age
4.Sex
5.Smoking status 
6.baseline week 


D. Lasso model :-
lasso also gave better cv results as compared to other machine learning models that i tried so  i used lasso for final model after tuning its hyper parametrs 
1. baseline_FVC
2.baseline_Percent 
3.Age
4.Sex
5.Smoking status 
6.baseline week 


E. Weighted Ensemble :

these are the final weights that i finally used after doing tons of subission .

final_model = efficientnet_model * 0.25 + qunatile_regressor * 0.44 + elatic_net * 0.18 + lasso * 0.13

# Things that didnt work:-
1. Used resnet instead of efficient net 
2. tried all the model version of the efficient net but b5 model was giving the best results 
3. used MAE for training of the effnet model
4. used normal pinball loss function in qunatile regressor model 
5. tried various macahine learning models such as lightgbm / ridge / bayesian ridge / ngboost / hubet / catboost 
6. added lstm layer in quantile regressor 
7. 3D efficient nets
8. Calucalted heights of each patient using the function that doctors used given by spirometry calculator ( also i took the average of the different race values )
9. Calculated FEV , fvc/fev ratio of each pateints similarly 
10. Image meta data didnt worked at all such as slice thickness 
11. tried to calculate the volume of lungs of patients but this feature also didnt helped .


# My opinion :-
I want to say that this was an awesome competetion that I am so glad I participated in! Very glad to get my first silver medal!

my kaggle profile link :- 
https://www.kaggle.com/trooperog

