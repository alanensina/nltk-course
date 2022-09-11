import nltk
from nltk import sent_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('vader_lexicon')

text = 'I love Galadriel the fighter. She is valiant, flawed and haughty, as bloody-minded as she is brilliant, scarred by the horrors of war. If that does not sound like much fun, wait till you see what she does to a snow troll. If the elves bring the intensity, then there is plenty of earthy light and joy in the harfoots, Tolkien predecessors to the hobbits, who are preparing for their seasonal migration. The young harfoots forage for berries and frolic in the mud, their elders (including Lenny Henry) on hand to explain how everything fits together, via some not-unwelcome exposition about who dwells where and what land they protect. The opening episode also introduces us to the Southlands, where elves and humans coexist uneasily amid decades of resentment in the aftermath of war.'

sent = sent_tokenize(text)
print(sent)

sa = SentimentIntensityAnalyzer()
for sentence in sent:
    print(sent)
    ps = sa.polarity_scores(sentence)
    for n in ps:
        print('{0}: {1}, '.format(n, ps[n]), end='')
    print()
    print()