import nltk
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk import ne_chunk, word_tokenize

sent = 'Traveling to Iceland is a great any time of year, whether you come for the eternal sun of summer or in search of winter is Northern Lights nature is sure to put on a show.'
print(ne_chunk(nltk.pos_tag(word_tokenize(sent)), binary=False))
