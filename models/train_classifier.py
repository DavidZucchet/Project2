import sys
# import libraries
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import nltk
nltk.download(['punkt', 'wordnet'])
import pickle

def load_data(database_filepath):

    #Returns the variables loaded for model process.

    #Parameters:
    #    database_filepath(str1):  string with the filepath of the SQL base.
    #Returns:
    #    X(df):  df with the messages
    #    y(df):  df with the 36 categories
    #    category_names(df):  df with columns names of the df

    # load messages dataset    
    
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("Messages", engine)
    X = df.message.values
    y = df.iloc[:, 4:].values
    return X, y, df.columns

def tokenize(text):
    
    #tokenization function to process and return text data

    #Parameters:
    #    text(str):  string text
    #Returns:
    #    clean_tokens(str):  string after tokenize applying lemmatizer, lower text and strip.

    
    #tokenization function to process your text data
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    #Returns a model applying pipeline with countvectorizer,tfidtransformer and multioutputclassifier with random forest. Also optimize parameters after using GridSearchCV

    #Returns:
    #    cv: model for fitting and predicting in next steps.
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'clf__estimator__min_samples_split': [2, 3,4]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    #function that evaluates a model with the testing data

    #Parameters:
    #    model:  model returned of the build_model function
    #    X(df):  df with the test messages
    #    y(df):  df with the test 36 categories
    #    category_names(df):  df with columns names of the df
    
    y_pred = model.predict(X_test)
    for i in range(y_pred.shape[1]):
        print(category_names[i])
        print(classification_report(Y_test[:,i],y_pred[:,i]))

def save_model(model, model_filepath):
    
    #function that saves the model as a pickle file

    #Parameters:
    #    model:  model returned of the evaluate_model function
    #    model_filepath(str):  string with the filepath of the pickle file

    #Returns:
    #    model as a pickle file.
    
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))

def main():
    #Function that performs model process for the messages data

    #Parameters:
    #    database_filepath(str1):  string with the filepath of the SQL base.
    #    model_filepath(str):  string with the filepath of the pickle file    
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()