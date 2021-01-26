import sys
# import libraries
import sqlite3
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    #Returns the files loaded for ETL process.

    #Parameters:
    #    messages_filepath (str):The string with the filepath of the csv messages
    #    categories_filepath (str):The string with the filepath of the csv categories
    #Returns:
    #    messages(df):  df with the messages 
    #    categories(df):  df with the categories
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    return messages,categories
def clean_data(messages,categories):
    #Returns the df with the merge of two inputs and the Data transformed.

    #Parameters:
    #    messages(df):  df with the messages 
    #    categories(df):  df with the categories
    #Returns:
    #    df: df merged with two inputs and cleaned. 
    
    # merge datasets
    df = messages.merge(categories,on='id')
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames=(row.apply(lambda x: x[0:-2])).values.tolist()
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # As "Related' field is more like "Indirectly Related", we convert 2 numbers as 1
        categories[column] = categories[column].apply(lambda x: 1 if x==2 else x)    
    # drop the original categories column from `df`
    df=df.drop(['categories'],axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,join='inner')
    
    df=df.drop_duplicates(subset=['message'])
    return df

def save_data(df, database_filename):
    
    #Function that saves the data in a SQL database

    #Parameters:
    #    df(df):  df with the transformed base
    #    database_filename(str1):  string with the filepath of the SQL base.

    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Messages', engine, index=False,if_exists='replace')


def main():
    #Function that performs ETL process for the messages data

    #Parameters:
    #    messages_filepath(df):   string with the filepath of the csv messages
    #    categories_filepath(str1):  string with the filepath of the csv categores
    #    database_filepath(str1):  string with the filepath of the SQL base.
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages,categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages,categories)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()