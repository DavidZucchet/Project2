# Disaster Response Pipeline Project

# Table of Contents
1. Installation
2. Project Motivation
3. File Descriptions
4. Results
5. Licensing, Authors, and Acknowledgements

# Installation
There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*.

# Project Motivation
For this project, I was interestested in using the data provides by Figure Eight to build a model for an API that classifies disastes messages and can send them to an appropriate disaster relief agency.

# File Descriptions
There is 3 folders containing the data, models and the app. 

The Data folder contains 2 CSV files with the raw data. The python file "process_data.py" has the code for extract, transform, clean and load the data.

The models folder contains a python file "train_classifier.py". When executing this code it splits the data into a training set and a test set. Then, it creates a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and RandomSearchCV (RandomizedSearchCV is used instead of GridSearchCV because sacrifice in precision and accuracy is not huge in relation to processing time and finding best parameters) to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). Finally, it exports the model to a pickle file.

The app folder contains a python file "run.py" that use the database and the model, and generate a web app using flask that displays data visualization and classify messages. It also has a folder with 2 html files for generating the web app. 
Comments were used to assist in walking through the thought process for individual steps.

# Results

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

The results of the model can be found at the web app.

# Licensing, Authors, Acknowledgements
Must give credit to Figure Eight for the data and Udacity for the learning path in the DataScientist Course and some of the code that I reused. Otherwise, feel free to use the code here as you would like!




