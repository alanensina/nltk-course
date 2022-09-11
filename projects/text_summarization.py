import nltk, wikipedia, heapq
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

wikipage = wikipedia.page('Artificial_intelligence').content
print(wikipage)

# Sentence tokenization
wiki_sent = sent_tokenize(wikipage)
print(wiki_sent)

# Word tokenization
wiki_words = []
for sent in wiki_sent:
    wiki_words.extend(word_tokenize(sent))
print(wiki_words)

# Word frequencies
stopwords = set(stopwords.words('english'))

word_frequencies = {}
for word in wiki_words:
    if word not in stopwords and word.isalpha():
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

print(word_frequencies)

# Calculating the weight of each word
max_frequency = max(word_frequencies.values())

for word in word_frequencies.keys():
    word_frequencies[word] = (word_frequencies[word]/max_frequency)

print(word_frequencies)

# Calculating the score of sentences
sentences_score = {}

for sent in wiki_sent:
    for word in wiki_words:
        if word.lower() in word_frequencies.keys():
            if len(sent.split(' ')) < 30:
                if sent not in sentences_score.keys():
                    sentences_score[sent] = word_frequencies[word]
                else:
                    sentences_score[sent] += word_frequencies[word]

print(sentences_score)

# Summarizing
summary_sentences = heapq.nlargest(7, sentences_score, key=sentences_score.get)
summary = ' '.join(summary_sentences)
print(summary)
