import joblib
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize

text_en = 'I was eating my dinner'
text_ptbr = 'Eu estava comendo meu jantar'

tokens_en = word_tokenize(text_en) # English by default
tokens_ptbr = word_tokenize(text_ptbr, language='portuguese') 

tags_en = nltk.pos_tag(tokens_en) # English by default

print('-----------------------------------')
print('English tags for: ' + text_en)
print(tags_en)
print('-----------------------------------')

# For portuguese language is not possible to tag using NLTK by default, NLTK accept only english and russian.
# I found a Github repository that implemmented a tagger to ptbr: https://github.com/inoueMashuu/POS-tagger-portuguese-nltk
# It is recommended to use POS_tagger_brill.pkl or POS_tagger_trigram.pkl, as they have the best accuracy and a satisfactory 
# rate of tagged words per second (Words/sec).
# -----------------------------------------------------------
# Tagger	                Accuracy	Words/sec	Size
# POS_tagger_trigram.pkl	85.19%	    61k	        2.05 MB
# POS_tagger_brill.pkl	    92.19%	    30k	        2.09 MB
# -----------------------------------------------------------

ptbr_tagger_brill = joblib.load('POS_tagger_brill.pkl')
ptbr_tagger_trigram = joblib.load('POS_tagger_trigram.pkl')

print('Portuguese tags for: ' + text_ptbr)
tags_ptbr = ptbr_tagger_brill.tag(tokens_ptbr)
print('Using brill as POS Tagger:')
print(tags_ptbr)
tags_ptbr = ptbr_tagger_trigram.tag(tokens_ptbr)
print('Using trigram as POS Tagger:')
print(tags_ptbr)
print('-----------------------------------')
