# Ročníkový projekt

## Knihovny

- TensorFlow - https://github.com/tensorflow/text/blob/master/docs/tutorials/text_classification_rnn.ipynb
- Scikit-learn + NLTK - https://realpython.com/python-nltk-sentiment-analysis/
- TextBlob - https://github.com/sloria/TextBlob
- VaderSentiment - https://github.com/cjhutto/vaderSentiment/blob/master/vaderSentiment/vaderSentiment.py, https://towardsdatascience.com/my-absolute-go-to-for-sentiment-analysis-textblob-3ac3a11d524

### Přístupy:

## Slovníkový

Spočítá kolik slov z textu je pozitivních a kolik negativních na základě předem napsaného slovníku.

## Naive Bayes

Vyžaduje předem označené texty.

Statistický algoritmus, který vynásobí pravděpodobnosti, že jsou slova pozitivní nebo negativní. (je nutné každě slovo jednou připsat do celého souboru dat, aby se nenásobilo nulou)

Text ohodnotí jako pozitivní nebo negativní podle toho, která pravděpodobnost je vyšší.

## Neuronové sítě

Vyžaduje předem označené texty.

https://developers.google.com/machine-learning/guides/text-classification/

1) Rozdělení textu (na slova, dvojice slov ...)
2) Převést slova na čísla (číslo nebo vektor)
3) Na sebe navazující vrstvy
4) Softmax funkce, která převede čísla na pravděpodobnosti

Diagramy neuronové sítě

https://towardsdatascience.com/how-to-easily-draw-neural-network-architecture-diagrams-a6b6138ed875
- YOLO v1 architecture
- VGG16 architecture

## Technologie

Python - programovací jazyk s největší podporou pro strojové učení a datovou analýzu