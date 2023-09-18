import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from joblib import dump, load
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.pipeline import Pipeline
from time import time
import numpy as np

def segundos_a_minutos(segundos):
    horas = int(segundos / 60 / 60)
    segundos -= horas*60*60
    minutos = int(segundos/60)
    segundos -= minutos*60
    return f"{horas:02d}:{minutos:02d}:{segundos:02d}"

model_cnb = load('./Trained Models/eng1.sav')

nltk.download('stopwords')

final_stopwords_list = stopwords.words('english')

json_file = './Datasets/News_Category_Dataset_v3.json'
df = pd.read_json(json_file, lines=True)

categories = ['ENTERTAINMENT','SCIENCE','POLITICS', 'PARENTING', 'TRAVEL'] #'BUSINESS', 'SPORTS','ARTS','ENVIRONMENT','EDUCATION','TECH']
df_news = df[df.category.isin(categories)]
df_news = df_news[['category','headline']]

vectorizer = TfidfVectorizer(stop_words=final_stopwords_list)
vectorizer.fit(df_news["headline"].values)
vect = vectorizer.transform(df_news["headline"].values)

vocab = np.sort(list(vectorizer.vocabulary_.keys()))
df_tf = pd.DataFrame(vect.todense(), columns = vocab)

x = df_tf.values
y = df_news["category"].values


train_df, test_df = train_test_split(df_news, test_size=0.2, random_state=42, stratify=df_news["category"])
train_data = train_df["headline"]
train_target = train_df["category"]
test_data = test_df["headline"]
test_target = test_df["category"]

nb_param_grid = {
    "reduce_dim": ["passthrough", TruncatedSVD(10), TruncatedSVD(20)],
    "tfidf__analyzer": ["word", "char"],
    "tfidf__smooth_idf": [True, False],
    "tfidf__ngram_range": [(1, 1), (1, 2),(2,2)],
    "tfidf__use_idf": [True, False],
    "tfidf__stop_words": [final_stopwords_list],
    "classifier__fit_prior": [True, False],
    "classifier__alpha": [0.1, 0.5, 1, 5, 10]
}

pipeline_CNB = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer()),
        ("reduce_dim", TruncatedSVD(n_components=10)),
        ("classifier", ComplementNB())
    ]
)

start = time()
print("Fitting started...")
search = RandomizedSearchCV(pipeline_CNB, 
                            param_distributions=nb_param_grid, 
                            verbose=1,
                            cv=5,
                            random_state=42)
search.fit(train_data, train_target)
end = time()
tiempo_train = segundos_a_minutos(int(end - start))
print(f"Tiempo de Entrenamiento {tiempo_train}")

model_cnb = search.best_estimator_

accuracy_train_cnb = round(model_cnb.score(train_data,train_target),5)
accuracy_test_cnb = round(model_cnb.score(test_data,test_target),5)

print(f'Train Accuracy : {accuracy_train_cnb}')
print(f'Test Accuracy  : {accuracy_test_cnb}')

y_pred = model_cnb.predict(test_data)
cr = classification_report(test_target, y_pred)
print(cr)

compare_model_cnb = pd.DataFrame({'test_data' : test_target, 
                                'prediction_data': y_pred
                              })

print(compare_model_cnb.sample(10))

filename = 'model_cnb.sav'
dump(model_cnb, filename)


