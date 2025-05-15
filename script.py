from pypdf import PdfReader
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import graphviz
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from re import sub as regexSubstitute
from cleantext import clean
#import cleantext
import io
import re
import string
import tqdm
from gensim.models import Word2Vec
import tensorflow as tf
#from tensorflow.keras import layers

#Add w2v, graph metrics to concepts, table of contents for full doc, w2v for abstract, remove references

#Converts POS tags to accepted format
def tagConversion(tag):
    tagsetMapping = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }

    start = tag[0].upper()
    tag = tagsetMapping.get(start)
    if tag == None: tag = wordnet.NOUN

    return tag

#Creates structures with frequency to get the graph
def freqArray(pairs):
    uniquePairs, counts = np.unique(pairs, axis = 0, return_counts = True)
    counts = counts.reshape(-1, 1)
    pairFreqs = np.concatenate((uniquePairs, counts), axis = 1)
    pairFreqs = np.array(sorted(pairFreqs, key=lambda x: -x[2]))

    return pairFreqs

#Calculates the cosine similarity between 2 vectors
def cosSim(a, b):
    res = np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))
    return res

def conMatrix(pairfreqs):
    dim = len(pairfreqs)
    matrix = np.zeros((dim, dim))
    for i in range(dim):
        matrix[pairfreqs[i][0], pairfreqs[i][1]] = pairfreqs[i][2]

    return matrix

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('universal_tagset')
nltk.download('stopwords')

#get the raw text from the pdf
reader = PdfReader("jstor-article.pdf")
corpus = ""
for page in reader.pages[1:]:
    raw = page.extract_text()
    # print(raw)
    raw = raw.split()
    #rint(raw)
    cutoff = 0
    for i in range(len(raw)):
        if raw[i] == "This" and raw[i + 1] == "content":
            cutoff = i
            break
    raw = raw[:cutoff]
    #print(raw)
    raw = " ".join(raw)
    corpus += raw
#print(corpus)
corpus = corpus[0:corpus.index("References")]

#print(corpus)
#remove <>, text that starts with www, special characters, first page of jstor pdf and use word2vec
#Clean up the text
#corpus = clean(corpus)
#     fix_unicode=True,               # fix various unicode errors
#     to_ascii=True,                  # transliterate to closest ASCII representation
#     lower=True,                     # lowercase text
#     no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
#     no_urls=True,                  # replace all URLs with a special token
#     no_emails=True,                # replace all email addresses with a special token
#     no_phone_numbers=True,         # replace all phone numbers with a special token
#     no_numbers=True,               # replace all numbers with a special token
#     no_digits=True,                # replace all digits with a special token
#     no_currency_symbols=True,      # replace all currency symbols with a special token
#     no_punct=True,                 # remove punctuations
# )

corpus = corpus.split()
corpus = [s for s in corpus if s[0] != '<' and s[-1] != '0' and len(s) > 2]
#print(corpus)
for i in range(len(corpus)):
    word = corpus[i]
    if word == '':
        continue
    if word[-1] == "-":
        word = word[:-1]
        word += corpus[i + 1]
        #print("joined", corpus[i], corpus[i + 1])
        #print(word)
        corpus[i] = word
        corpus[i + 1] = ''
corpus = ' '.join(corpus)
corpus = corpus.replace("-", " ")

#re.sub(r'\((?:[\w \.&]+\, )+[0-9]{4}\)', "", corpus)
#with open("OutputClean.txt", "w") as text_file:
    #text_file.write(corpus)
#print(corpus)

#Tokenization
corpus = unidecode(corpus.lower())
corpus = regexSubstitute(r'[^a-z\s]', '', corpus)
#print(corpus)
corpus = regexSubstitute(r'\s+', ' ', corpus).split()
# corpus = corpus[0:corpus.index("references")]
#print(corpus)

stopWords = set(stopwords.words('english'))
stopWords.update(['us','whose'])
tokenCorpus = word_tokenize(str.join(" ",[x for x in corpus if x not in stopWords and len(x) >= 2]))
print(tokenCorpus)

#Tagging and Lemmatization
patternTags = nltk.pos_tag(tokenCorpus)

lemmatizer = WordNetLemmatizer()
lemmas = []
vocab = set()
for token, tag in patternTags:
    lemma = lemmatizer.lemmatize(token, pos = tagConversion(tag))
    lemmas.append(lemma)
    vocab.add(lemma)

#Frequency data and graph creation
frequency = FreqDist(tokenCorpus)

wordIndex = {}
#indexWord = {0: '<pad>'}
indexWord = {}
index = 0
checked = []
for word in tokenCorpus:
    if word not in checked:
        wordIndex[word] = index
        indexWord[index] = word
        index += 1
        checked.append(word)
print(wordIndex)
# print(len(wordIndex))
seq = [wordIndex[word] for word in tokenCorpus]
#print("seq", seq)
# print(tokenCorpus)
ngr = ngrams(tokenCorpus, 2)
ngrList = [list(x) for x in list(ngr)]
#print(list(ngr))
#print(ngrList)
w2v = Word2Vec(ngrList, min_count=1)
#w2v.build_vocab(ngr)
#print(len(w2v.wv))
totalLen = len(wordIndex)
for i in range(totalLen):
    #print(indexWord[i], w2v.wv[i])
    pass

# grid = [[0 for j in range(totalLen)] for i in range(totalLen)]
# for i in range(totalLen):
#     for j in range(i, totalLen):
#         grid[i][j] = grid[j][i] = cosSim(w2v.wv[i], w2v.wv[j])
#print(grid)

wordContexts = []
for i in range(len(tokenCorpus)):
    #print(tokenCorpus[i])
    #print(cosSim(w2v.wv[i], w2v.wv[i + 1]))
    context = []
    if i > 2 and i < len(tokenCorpus) - 2:
        context = [tokenCorpus[i - 2], tokenCorpus[i - 1], tokenCorpus[i + 1], tokenCorpus[i + 2]]
    elif i < 2:
        context = [tokenCorpus[i + 1], tokenCorpus[i + 2]]
    else:
        context = [tokenCorpus[i - 1], tokenCorpus[i - 2]]
    # print(context)
    agg = w2v.wv[wordIndex[context[0]]].copy()
    #print(agg)
    for word in context[1:]:
        for j in range(len(agg)):
            agg[j] += w2v.wv[wordIndex[word]][j]
    # print(context)
    # print(agg)
    wordContexts.append((tokenCorpus[i], agg))
# print(len(wordContexts))

grid = []
for i in range(len(wordContexts)):
    for j in range(i, len(wordContexts)):
        word1 = wordContexts[i][0]
        word2 = wordContexts[j][0]
        if word1 != word2:
            wordSim = cosSim(wordContexts[i][1], wordContexts[j][1])
            if wordSim >= 0.4:
                grid.append([word1, word2, wordSim])

#Graph Creation
# window = 2
# indexPairs = []
# for centerPos in range(len(lemmas)):
#     for w in range(-window, window + 1):
#         contextPos = centerPos + w
#         if contextPos in range(len(lemmas)) and centerPos != contextPos:
#             contextIndex = wordIndex[lemmas[contextPos]]
#             centerIndex = wordIndex[lemmas[centerPos]]
#             indexPairs.append((centerIndex, contextIndex))

# freqPairs = freqArray(indexPairs)
# print("pairing done")

# np.savetxt('pairFrequency.csv',  freqPairs, delimiter=',', fmt='%d')

# matrix = conMatrix(freqPairs)

dataframe = pd.DataFrame(grid)
dataframe.columns = ['A', 'B', 'Weight']
print(dataframe)

dot = graphviz.Graph('Word Graph', engine='neato')
graph = nx.Graph()
print("graph done")

# Create the data frame
for _, row in dataframe.iterrows():
    a = row['A']
    b = row['B']
    w = row['Weight']
    print(a, b, w)
    graph.add_edge(a, b, weight = w)

    dot.node(str(row['A']), row['A'])
    dot.node(str(row['B']), row['B'])
    dot.edge(str(row['A']), str(row['B']), weight= str(w))

#Export graph data
dot.render(filename="./wordgraph.gv")
print("dataframe done")

positions = nx.spring_layout(graph)
weights = nx.get_edge_attributes(graph, 'weight')

print("maybe here")
for node in graph.nodes:
    shortest_path = nx.shortest_path(graph, source=node, weight='weight')
    degree_centrality = nx.degree_centrality(graph)
    betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')
    closeness_centrality = nx.closeness_centrality(graph, distance='weight')
    clustering_coefficient = nx.clustering(graph, weight='weight')
print("maybe not")

df_metrics = pd.DataFrame(indexWord.items(), columns=['Node', 'word'])
df_metrics['frequency_word'] = df_metrics['word'].map(frequency)
df_metrics['shortest_path'] = df_metrics['Node'].map(shortest_path)
df_metrics['degree_centrality'] = df_metrics['Node'].map(degree_centrality)
df_metrics['betweenness_centrality'] = df_metrics['Node'].map(betweenness_centrality)
df_metrics['closeness_centrality'] = df_metrics['Node'].map(closeness_centrality)
df_metrics['clustering_coefficient'] = df_metrics['Node'].map(clustering_coefficient)
df_metrics['eigenvector_centrality'] = df_metrics['Node'].map(nx.eigenvector_centrality(graph, weight='weight'))
df_metrics = df_metrics.sort_values("eigenvector_centrality", ascending=False)

def class_quartiles(df,name):
    name1 = 'Class_' + name + 'low'
    name2 = 'Class_' + name + 'medium'
    name3 = 'Class_' + name + 'high'
    quartiles = np.percentile(df[name], [25, 50, 75])
    df[name1] = np.where(df[name] >= quartiles[2], 1, 0)
    df[name2] = np.where((df[name] > quartiles[0]) & (df[name] < quartiles[2]), 1, 0)
    df[name3] = np.where((df[name] <= quartiles[0]) | (df[name] >= quartiles[2]), 1, 0)
    return df

df_metrics=class_quartiles(df_metrics,'degree_centrality')
df_metrics=class_quartiles(df_metrics,'betweenness_centrality')
df_metrics=class_quartiles(df_metrics,'closeness_centrality')
df_metrics=class_quartiles(df_metrics,'clustering_coefficient')
df_metrics=class_quartiles(df_metrics,'eigenvector_centrality')

## save the metrics in a csv file
df_metrics.to_csv('metrics.csv', index=False)

## create the dataframe only with the word and all columns beging Class

df_metrics_class=df_metrics.filter(regex='Class', axis=1)

# df_metric aggregate the metrics of each word in THE FIRST COLUMN
df_metrics_class['word']=df_metrics['word']

columnas = df_metrics_class.columns.tolist()
columnas = [columnas[-1]] + columnas[:-1]
df_metrics_class = df_metrics_class[columnas]

# save the dataframe in a csv file
df_metrics_class.to_csv('metrics_class.csv', index=False)
print("metrics done")
