#run by line 
#/Users/Lance/opt/anaconda3/envs/MLEnv/bin/python -i

import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics
import ktrain
from ktrain import text

DATA_PATH = '/Users/Lance/Developer/MachineLearning/new_train.csv'
NUM_WORDS = 50000
MAXLEN = 150
categories = ['Social disposition']
(x_train, y_train), (x_test, y_test), preproc = text.texts_from_csv(DATA_PATH,
                      'Utterance',
                      label_columns = categories,
                      val_filepath=None, # if None, 10% of data will be used for validation
                      max_features=NUM_WORDS, maxlen=MAXLEN,
                      ngram_range=1)

model = text.text_classifier('fasttext', (x_train, y_train) , preproc=preproc)
learner = ktrain.get_learner(model, 
                             train_data=(x_train, y_train), 
                             val_data=(x_test, y_test), 
                             batch_size=6)

learner.autofit(0.005)

learner.view_top_losses(n=4,preproc=preproc)

predictor = ktrain.get_predictor(learner.model, preproc)

predictor.predict_proba("Oh so I think that's really great uh any other comments on the video?")

from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=categories)
exp = explainer.explain_instance("That's really cool. Would you mind typing the name in the group chat so we can look him up afterwards?",predictor.predict_proba, num_features= 6)
exp.save_to_file('/Users/Lance/Developer/MachineLearning/oi.html')


#Saving
predictor.save('/Users/Lance/Developer/MachineLearning/NewSD')
#reloaded_predictor = ktrain.load_predictor('/Users/Lance/Developer/MachineLearning/Predictors')

#apply
import pandas as pd
import numpy as np
data = pd.read_csv("/Users/Lance/Developer/MachineLearning/train.csv")
pred_ai = []
for n in range(0, len(data.index)):
  pred_ai.append((predictor.predict_proba(data['Utterance'][n])[1]))
data['pred_SocialDisposition'] = pred_ai
data.to_excel('output1.xlsx') 