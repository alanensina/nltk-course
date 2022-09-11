import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, UnigramTagger, BigramTagger, TrigramTagger, DefaultTagger
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
#print(brown_tagged_sents)

# We are dividing the data into a test and train to evaluate our taggers
train_data = brown_tagged_sents[:int(len(brown_tagged_sents) * 0.9)] # [start : end]
test_data = brown_tagged_sents[int(len(brown_tagged_sents) * 0.9):]

unigram_tagger = UnigramTagger(train_data)
print('Accuracy of Unigram on this corpus without backoff:')
print(unigram_tagger.accuracy(test_data)) # evaluate method is deprecated, so, it was replaced to accuracy

default_tagger = DefaultTagger('NN') # Define that all words are nouns

# The backoff means, if the tagger doesn't know which tag to put to a word, they put the tagger defined on the backoff
unigram_tagger = UnigramTagger(train_data, backoff=default_tagger)
print('Accuracy of Unigram on this corpus assuming DefaultTagger as backoff:')
print(unigram_tagger.accuracy(test_data))

bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)
print('Accuracy of Bigram on this corpus assuming UnigramTagger as backoff:')
print(bigram_tagger.accuracy(test_data))

trigram_tagger = TrigramTagger(train_data, backoff=bigram_tagger)
print('Accuracy of Trigram on this corpus assuming BigramTagger as backoff:')
print(trigram_tagger.accuracy(test_data))
print('-------------------------------------------------')

text = "Iceland is a captivating destination for any traveler and with a host of low cost flight options from North America and mainland Europe to the land of ice and fire there has never been a better time to visit Iceland. Volcanoes, glaciers, the wind and the sea merge to create a landscape that is like nowhere else on Earth."

print(text)
print('-------------------------------------------------')

sent = nltk.sent_tokenize(text) # tokenize the sentences
tokens = [] # create an empty list
for s in sent:
    tokens.extend(word_tokenize(s))
print('Tokens:')
print(tokens)
print('-------------------------------------------------')
print('Tags using BigramTagger:')
print(bigram_tagger.tag(tokens))

