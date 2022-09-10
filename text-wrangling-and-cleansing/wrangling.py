import urllib.request
from bs4 import BeautifulSoup

url = "https://www.letras.mus.br/led-zeppelin/68345/"
html = urllib.request.urlopen(url).read()
#print(html)

raw = BeautifulSoup(html, 'html.parser').get_text()
print(raw)
