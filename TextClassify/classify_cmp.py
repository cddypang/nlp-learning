# encoding: utf-8

from transformers import *

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import random


class EmptyTransformer():
    def __init__(self):
        pass

    def fit(self, df_x, df_y=None):
        # print("do fit ...")
        return self

    def transform(self, in_x):
        # print("do transform ...")
        return in_x


if __name__ == "__main__":
    in_x = []
    in_y = []
    with open('news_train.txt','r',encoding='utf-8') as f:
        for l in f.readlines():
            l = l.replace('\n','')
            in_x.append(l.split('\t')[0])
            in_y.append(l.split('\t')[1].replace('__label__',''))

    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(in_x)
    random.seed(randnum)
    random.shuffle(in_y)

    doc2vec_trf = Doc2VecTransformer()
    doc2vec_features = doc2vec_trf.fit(in_x).transform(in_x)
    # print(doc2vec_features)
    print("doc2vec vector shape: ", doc2vec_features.shape)

    # pl_log_reg = Pipeline(steps=[('doc2vec',Doc2VecTransformer()),
    #     ('log_reg', LogisticRegression(multi_class='auto', solver='liblinear', max_iter=100))])
    #
    # scores = cross_val_score(pl_log_reg, in_x, in_y, cv=5,scoring='accuracy')
    # print('Accuracy for Logistic Regression: ', scores.mean())

    classes = list(set(in_y))
    label_y = []
    output_empty = [0] * len(classes)
    for y in in_y:
        output_row= list(output_empty)
        output_row[classes.index(y)]= 1
        label_y.append(output_row)

    test_x = []
    test_y = []
    with open('news_test.txt','r',encoding='utf-8') as f:
        for l in f.readlines():
            l = l.replace('\n','')
            test_x.append(l.split('\t')[0])
            test_y.append(l.split('\t')[1].replace('__label__',''))

    randnum = random.randint(0, 100)
    random.seed(randnum)
    random.shuffle(test_x)
    random.seed(randnum)
    random.shuffle(test_y)

    test_doc2vec_features = doc2vec_trf.transform(test_x)
    print("test_doc2vec vector shape: ", test_doc2vec_features.shape)

    test_label_y = []
    for y in test_y:
        output_row= list(output_empty)
        output_row[classes.index(y)]= 1
        test_label_y.append(output_row)

    pl_log_reg = Pipeline(steps=[('doc2vec',EmptyTransformer()),
        ('log_reg', LogisticRegression(multi_class='auto', solver='liblinear', max_iter=100))])
    scores = cross_val_score(pl_log_reg, doc2vec_features, in_y, cv=5,scoring='accuracy')
    print('Accuracy for Logistic Regression Classifier : ', scores.mean())

    # pl_random_forest = Pipeline(steps=[('doc2vec',EmptyTransformer()),
    #     ('random_forest',RandomForestClassifier())])
    # scores = cross_val_score(pl_random_forest, doc2vec_features, in_y, cv=5, scoring='accuracy')
    # print('Accuracy for RandomForest Classifier : ', scores.mean())
    #
    # pl_xgb = Pipeline(steps=[('doc2vec',EmptyTransformer()),
    #     ('xgb_boost', xgb.XGBClassifier(objective='binary:logistic'))])
    # scores = cross_val_score(pl_xgb, doc2vec_features, in_y, cv=5)
    # print('Accuracy for XGBoost Classifier : ', scores.mean())

    log_reg = LogisticRegression(multi_class='auto',solver='liblinear',max_iter=100)
    log_reg.fit(doc2vec_features, in_y)
    # predict_y = log_reg.predict(test_doc2vec_features)
    print('Accuracy for Logistic Regression Classifier test : ',
          log_reg.score(test_doc2vec_features, test_y))

    tfidf_transformer = Text2TfIdfTransformer()
    tfidf_vectors = tfidf_transformer.fit(in_x).transform(in_x)
    print("tf-idf vector shape: ", tfidf_vectors.shape)

    test_tfidf_features = tfidf_transformer.transform(test_x)
    print("test_tf-idf vector shape: ", test_tfidf_features.shape)

    pl_log_reg_tf_idf = Pipeline(steps=[('tfidf', EmptyTransformer()),
        ('log_reg', LogisticRegression(multi_class='auto', solver='liblinear', max_iter=100))])
    scores = cross_val_score(pl_log_reg_tf_idf, tfidf_vectors, in_y, cv=5, scoring='accuracy')
    print('Accuracy for Tf-Idf & Logistic Regression: ', scores.mean())

    # pl_random_forest_tf_idf = Pipeline(steps=[('tfidf', EmptyTransformer()),
    #                             ('random_forest', RandomForestClassifier())])
    # scores = cross_val_score(pl_random_forest_tf_idf, tfidf_vectors, in_y, cv=5, scoring='accuracy')
    # print('Accuracy for Tf-Idf & RandomForest : ', scores.mean())
    #
    # pl_xgb_tf_idf = Pipeline(steps=[('tfidf', EmptyTransformer()),
    #                         ('xgboost', xgb.XGBClassifier(objective='binary:logistic'))])
    # scores = cross_val_score(pl_xgb_tf_idf, tfidf_vectors, in_y, cv=5)
    # print('Accuracy for Tf-Idf & XGBoost Classifier : ', scores.mean())

    log_reg.fit(tfidf_vectors, in_y)
    print('Accuracy for Tf-Idf & Logistic Regression test : ',
          log_reg.score(test_tfidf_features, test_y))

