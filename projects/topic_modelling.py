import nltk, wikipedia
import pyLDAvis.gensim_models
from gensim import corpora, models
from nltk import sent_tokenize, word_tokenize, WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords, wordnet

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

wikipage = wikipedia.page('Coronavirus').content
print(wikipage)

# Sentence tokenization
wiki_sent = sent_tokenize(wikipage)
print(wiki_sent)

# Word tokenization
wiki_words = []
for sent in wiki_sent:
    wiki_words.extend(word_tokenize(sent))
print(wiki_words)

# POS Tagging
tagged = pos_tag(wiki_words)
print(tagged)

# Function to use POS Tags on Lemmatizer
def getpos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''
    
# Lemmatization with POS
punctuation = u",.?!()-_\"\'\\\n\r\t;:+*<>@#§^$%&|/"
stop_words_eng = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()
lemma_pos = []
for word, tag in tagged:
  if word not in punctuation and word not in stop_words_eng and word.isalpha():
    p = getpos(tag)
    if p != '':
      l = wordnet_lemmatizer.lemmatize(word, pos = p)
      lemma_pos.append(l)
print(lemma_pos)

id2word = corpora.Dictionary([lemma_pos])
texts = lemma_pos
corpus = [id2word.doc2bow([text]) for text in texts]

n_topics = 5
lda_model = models.LdaModel(corpus=corpus,
                            id2word=id2word,
                            num_topics=5,
                            random_state=100,
                            update_every=1,
                            chunksize=100,
                            passes=10,
                            alpha='symmetric',
                            per_word_topics=True)

print(lda_model.print_topics())

vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
print(vis)