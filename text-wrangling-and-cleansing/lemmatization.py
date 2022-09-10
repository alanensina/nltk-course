import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

snw_ptbr = SnowballStemmer('portuguese')
snw_en = SnowballStemmer('english')

words_ptbr = ['comer', 'andando', 'jogado', 'revisto', 'falar', 'rato']
words_en = ['eat', 'walking', 'played', 'revised', 'talk', 'mice']

print('English words stemmed:')
for word_en in words_en:
    print(word_en + '---> ' + snw_en.stem(word_en))
    
print('Portuguese words stemmed:')
for word_pt in words_ptbr:
    print(word_pt + '---> ' + snw_ptbr.stem(word_pt))
    
# Lemmatization is the grouping together of different forms of the same word. 
# Example: mice == mouse
wlem = WordNetLemmatizer()
word_lemmatized = wlem.lemmatize('mice')
print('mice lemmatized: ' + word_lemmatized)