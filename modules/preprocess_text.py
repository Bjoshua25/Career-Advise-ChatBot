# ========== Import Libraries =========
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# load stop words
stopwords = set(stopwords.words('english'))

# Instantiate the lemmatizer
lemmarizer = WordNetLemmatizer()


# ======== Tokenizer Function ========
def tokenize_text(text):
    """This function tokenizes the input text, removes stop words, and lemmatizes the tokens.
    Args:
        text (str): The input text to be tokenized.

    Returns:
        list: A list of lemmatized tokens."""
    
    
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stopwords]
    lemmatized_tokens = [lemmarizer.lemmatize(word) for word in filtered_tokens if word.isalpha()]
    return lemmatized_tokens