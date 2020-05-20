import pandas as pd
import numpy as np
import nltk
import sklearn
from sklearn.datasets import load_files
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

code = "Content_Focus"

df = pd.read_csv('x5set.csv')
raw_data = df[['Utterance',code]]
X = raw_data['Utterance']
y = raw_data[code]
print(code +" Distribution:\n",raw_data[code].value_counts())

from text_prep import text_prep

# Text preprocessing
documents = text_prep(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(documents, y, test_size=0.1, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X_train = tfidfconverter.fit_transform(X_train)
X_test = tfidfconverter.transform(X_test)

#Smote
from imblearn.over_sampling import SMOTE
from collections import Counter
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
print(sorted(Counter(y_resampled).items()))


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
rfc = RandomForestClassifier()
#train the algorithm on training data and predict using the testing data
pred = rfc.fit(X_resampled, y_resampled).predict(X_test)
model = rfc.fit(X_resampled,y_resampled)
#print the accuracy score of the model
print("Random Forest Classifier Accuracy: ",accuracy_score(y_test, pred, normalize = True))
print("Random Forest Classifier Recall: ",recall_score(y_test, pred, average='weighted'))

#apply
import pandas as pd
import numpy as np
data = pd.read_csv("/Users/Lance/Developer/MachineLearning/train.csv")
pred_ai = []
prep_data = text_prep(data['Utterance'])
X = tfidfconverter.transform(prep_data).toarray()
predicted = model.predict_proba(X)
for n in range(0, len(data.index)):
  pred_ai.append((predicted[n][1]))

data['pred_'+code] = pred_ai

new_data = data[['Utterance',code,'pred_'+code]]
new_data.to_excel(code+'_test2.xlsx') 


#explain 
# from sklearn.pipeline import make_pipeline
# c = make_pipeline(tfidfconverter,rfc)

# from lime import lime_text
# from lime.lime_text import LimeTextExplainer
# explainer = LimeTextExplainer(class_names=['no','yes'])

# idx = 60
# exp = explainer.explain_instance(documents[idx], c.predict_proba, num_features=5)
# exp.save_to_file('/Users/Lance/Developer/MachineLearning/oi.html')

