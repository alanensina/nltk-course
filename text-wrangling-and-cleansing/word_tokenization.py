from nltk.tokenize import word_tokenize, WordPunctTokenizer

text_en = "I just can't get enough!"

text_ptbr = "Aquela atitude foi a gota d'água!"

en_tokens = word_tokenize(text_en)
ptbr_tokens = word_tokenize(text_ptbr, language='portuguese')

# Examples without punctuation
print('Examples without punctuation')
print('English tokens:')
print(en_tokens)

print('Portuguese tokens:')
print(ptbr_tokens)

# Examples with punctuation
print('Examples with punctuation')
tokenizer = WordPunctTokenizer()
en_tokens_cleaned = tokenizer.tokenize(text_en)
ptbr_tokens_cleaned = tokenizer.tokenize(text_ptbr)
print('English tokens:')
print(en_tokens_cleaned)
print('Portuguese tokens:')
print(ptbr_tokens_cleaned)
