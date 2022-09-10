import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop_list_pt = stopwords.words('portuguese')
stop_list_en = stopwords.words('english')

print('------------------------------')
print('All words in English into stopwords:')
print(stop_list_en)
print('------------------------------')
print('All words in Portuguese into stopwords:')
print(stop_list_pt)
print('------------------------------')
print('List of words in English without stop words:')
text_en = "This is just a test"
clean_word_list_en = [word for word in text_en.split() if word not in stop_list_en]
print(clean_word_list_en)
print('------------------------------')
print('List of words in Portuguese without stop words:')
text_pt = "Isso é apenas um simples teste"
clean_word_list_pt = [word for word in text_pt.split() if word not in stop_list_pt]
print(clean_word_list_pt)
print('------------------------------')