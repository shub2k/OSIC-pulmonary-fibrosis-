# OSIC-pulmonary-fibrosis-Competetion rank - 67 ( silver medal ) My score = -6.847 , First position score = -6.8305
  https://www.kaggle.com/c/osic-pulmonary-fibrosis-progression

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

A. efficientnet with b5 layer which were trained on  windowed lung ct scan images along with the meta data for 5 folds. 
the approach on this model was to use the images to predict the betas (slopes for the FVC declines ) 
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

D. Lasso model :-
lasso also gave better cv results as compared to other machine learning models that i tried so  i used lasso for final model after tuning its hyper parametrs 

E. Weighted Ensemble :

these are the final weights that i finally used after doing tons of subission .

final_model = efficientnet_model * 0.25 + qunatile_regressor * 0.44 + elatic_net * 0.18 + lasso * 0.13


