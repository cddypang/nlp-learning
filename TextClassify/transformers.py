# encoding: utf-8

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.base import BaseEstimator
from sklearn import utils as skl_utils
from tqdm import tqdm
import multiprocessing
import numpy as np

class Doc2VecTransformer(BaseEstimator):
    def __init__(self, vector_size=100, learning_rate=0.02, epochs=5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self._model = None
        self.vector_size = vector_size
        self.workers = multiprocessing.cpu_count() - 1

    def fit(self, df_x, df_y=None):
        tagged_x = [TaggedDocument(row.split(), [index]) for index, row in enumerate(df_x)]
        model = Doc2Vec(documents=tagged_x, vector_size=self.vector_size, workers=self.workers)
        for epoch in range(self.epochs):
            model.train(skl_utils.shuffle([x for x in tqdm(tagged_x)]), total_examples=len(tagged_x), epochs=1)
            model.alpha -= self.learning_rate
            model.min_alpha = model.alpha
            self._model = model

        return self

    def transform(self, df_x):
        return np.asmatrix(np.array([self._model.infer_vector(x.split()) for x in df_x]))


# tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer

class Text2TfIdfTransformer(BaseEstimator):
    def __init__(self):
        self._model = TfidfVectorizer()
        pass

    def fit(self, df_x, df_y=None):
        # df_x = df_x.apply(lambda x : clean_text(x))
        self._model.fit(df_x)

        return self

    def transform(self, df_x):
        return self._model.transform(df_x)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


if __name__ == "__main__":
    in_x = []
    in_y = []
    with open('news_train.txt','r',encoding='utf-8') as f:
        for l in f.readlines():
            l = l.replace('\n','')
            in_x.append(l.split('\t')[0])
            in_y.append(l.split('\t')[1].replace('__label__',''))

    # doc2vec_trf = Doc2VecTransformer()
    # doc2vec_features = doc2vec_trf.fit(in_x).transform(in_x)
    # print(doc2vec_features)
    #
    # pl_log_reg = Pipeline(steps=[('doc2vec',Doc2VecTransformer()),
    #     ('log_reg', LogisticRegression(multi_class='auto', solver='liblinear', max_iter=100))])
    #
    # scores = cross_val_score(pl_log_reg, in_x, in_y, cv=5,scoring='accuracy')
    # print('Accuracy for Logistic Regression: ', scores.mean())

    tfidf_transformer = Text2TfIdfTransformer()
    tfidf_vectors = tfidf_transformer.fit(in_x).transform(in_x)
    print("tf-idf vector shape: ", tfidf_vectors.shape)

    pl_log_reg_tf_idf = Pipeline(steps=[('tfidf', Text2TfIdfTransformer()),
        ('log_reg', LogisticRegression(multi_class='auto', solver='liblinear', max_iter=100))])
    scores = cross_val_score(pl_log_reg_tf_idf, in_x, in_y, cv=5, scoring='accuracy')
    print('Accuracy for Tf-Idf & Logistic Regression: ', scores.mean())

