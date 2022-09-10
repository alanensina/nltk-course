# The Stemmers are used to get the root of the word, so, the root of the verb eat is eat, because when you conjugate the verb can be: eating, eaten, ate
# There are 3 different stemmers:
# The Porter Stemmer, good enough to English, it's simple and fast
# LancasterStemmer was developed in 1990 and uses more aggressive approch than The Porter Stemmer Algorithm
# Snowball stemmers can be used for 13 languages. English and Portuguese included.
# There's a specific stemmer for portuguese language: RSLPStemmer

import nltk
nltk.download('rslp')
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.rslp import RSLPStemmer

pst = PorterStemmer()
lst = LancasterStemmer()
snw_ptbr = SnowballStemmer('portuguese')
snw_en = SnowballStemmer('english')
rslp = RSLPStemmer()

word_stemmed_lst = lst.stem('eating')
word_stemmed_pst = pst.stem('eating')
word_stemmed_snw_en = snw_en.stem('eating')
word_stemmed_snw_ptbr = snw_ptbr.stem('comendo')
word_stemmed_rslp = rslp.stem('comendo')

print('Porter Stemmer for eating: ' + word_stemmed_pst)
print('LancasterStemmer for eating: ' + word_stemmed_lst)
print('SnowballStemmer for eating: ' + word_stemmed_snw_en)
print('SnowballStemmer for eating in portuguese: ' + word_stemmed_snw_ptbr)
print('RSLPStemmer for eating in portuguese: ' + word_stemmed_rslp)