import os
import logging
import string
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')

#Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#Setting up logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transforms the input text by converting it to lowercase,tokenizing removing stopwords and punctuation, and stemming"""
    ps = PorterStemmer()
    tokenizer = TreebankWordTokenizer()

    #Convert to lowercase
    text = text.lower()

    #Tokenize the text
    text = tokenizer.tokenize(text)

    #Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]

    #Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

    #Stem the words
    text = [ps.stem(word) for word in text]

    #Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    """
    Preprocess the DataFrame by encoding the target column,removing duplicates,ans transforming the text column
    """
    try:
        logger.debug('Starting preprocessing for DataFrame')
        #Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        #Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')

        #Apply text transformation to the specified text column
        df.loc[:,text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df

    except KeyError as e:
        logger.error('Column not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error during text normaliztion: %s',e)
        raise

def main(text_column='text',target_column='target'):
    """
    Main function to load raw data,preprcoess it,and save the processed data
    """
    try:
        #Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        #Transform the data
        train_processed_data =preprocess_df(train_data,text_column,target_column)
        test_processed_data = preprocess_df(test_data,text_column,target_column)

        #Store the data inside data/processed
        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path,'train_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test.processed.csv'),index=False)

        logger.debug('Processed data saved to: %s',data_path)

    except FileNotFoundError as e:
        logger.error('FIle not found: %s',e)

    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s',e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s',e)
        print(f"Error:{e}")

if __name__ == '__main__':
    main()
