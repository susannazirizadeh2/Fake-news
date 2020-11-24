from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize 
import nltk
import re
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


nltk.download('stopwords')



# Interface lemma tokenizer from nltk with sklearn
class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return[self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]
    
stop_words = set(stopwords.words('english')) 
# Lemmatize the stop words
tokenizer=LemmaTokenizer()
token_stop = tokenizer(' '.join(stop_words))

def make_features():
	text_features_title = 'title'
	text_transformer_title = Pipeline(steps=[
	    ('cnt', TfidfVectorizer(tokenizer=tokenizer,
	                            analyzer='word',
	                            ngram_range=(1,2),
	                            max_features=100,
	                            stop_words=token_stop,
	                            token_pattern='[A-Za-z][\w\-]*',
	                            max_df=0.25,
	                            use_idf=True,
	                            lowercase=True))
	])


	text_features_text = 'text'
	text_transformer_text = Pipeline(steps=[
	    ('cnt', TfidfVectorizer(tokenizer=tokenizer,
	                            analyzer='word',
	                            ngram_range=(1,2),
	                            max_features=2000,
	                            stop_words=token_stop,
	                            token_pattern='[A-Za-z][\w\-]*',
	                            max_df=0.25,
	                            use_idf=True,
	                            lowercase=True))
	])

	preprocessor = ColumnTransformer(transformers=[
	    ('txt_text', text_transformer_text, text_features_text),
	    ('txt_title', text_transformer_title, text_features_title),
	])

	estimators = [('preprocessor', preprocessor),
	              ('classifier', LogisticRegression(C=10, penalty='l2',solver='lbfgs'))]
	pipe = Pipeline(estimators)
	return pipe