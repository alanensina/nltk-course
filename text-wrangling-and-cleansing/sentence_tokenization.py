import nltk
import nltk.data
nltk.download('punkt')
nltk.download('webtext')
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import webtext

text_en = "Good-bye! No, do not grieve that it is over. The perfect hour. That the winged joy, sweet honey-loving rover, flits from the flower. Grieve not. It is the law. Love will be flying. Yes, love and all. Glad was the living, blessed be the dying. Let the leaves fall."

text_ptbr = "Sofrer por você, viver lutar. De joelhos hesitar, por não te desejar. Passos largos em direção, desejos por trás de desejos, por tanto te odiar. E por isso cantar, por isso trabalhar, por isso roubar. Por mais forte que seja, você está aqui, dinheiro."

text_web = webtext.raw('overheard.txt')

sentences_en = sent_tokenize(text_en) # English by default
sentences_ptbr = sent_tokenize(text_ptbr, language='portuguese')
print('English sentences:')
print(sentences_en)
print('Portuguese sentences:')
print(sentences_ptbr)

# Different way to tokenize:
#pt_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
#sentences_ptbr = pt_tokenizer.tokenize(text_ptbr)
#print(sentences_ptbr)

# Tokenize by a web text and train a model:
#print(text_web)
sent_tokenizer = PunktSentenceTokenizer(text_web)
sentences_web_with_train = sent_tokenizer.tokenize(text_web)

# Print only the first 5 sentence
for x in range(5):
    print(sentences_web_with_train[x])  
    
# Difference between a train tokenizer and a standard one:
sentences_web_without_train = sent_tokenize(text_web)
sent_with_train = sentences_web_with_train[678]
sent_without_train = sentences_web_without_train[678]
print('--- Difference between a train tokenizer and a standard one: ---')
print('With train: ')
print(sent_with_train) # Note, only one sentence is printed
print('Without train: ')
print(sent_without_train) # Note, two sentences are printed
print('------')