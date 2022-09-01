import pandas as pd
import re
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.decomposition import TruncatedSVD
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')
data1 = pd.read_csv("Indiegogo001.csv")
data2 = pd.read_csv("Indiegogo002.csv")
data3 = pd.read_csv("Indiegogo003.csv")
data4 = pd.read_csv("Indiegogo004.csv")
data5 = pd.read_csv("Indiegogo005.csv")
document1 = data1['title'].str.cat(sep = ' ')
document2 = data2['title'].str.cat(sep = ' ')
document3 = data3['title'].str.cat(sep = ' ')
document4 = data4['title'].str.cat(sep = ' ')
document5 = data5['title'].str.cat(sep = ' ')
totdoc = [document1, document2, document3, document4, document5]
# list for tokenized documents in loop
texts = []
#Tokenizer
tokenizer = RegexpTokenizer(r'\w+')
# create English stop words list
en_stop = get_stop_words('en')
# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
# loop through document list
for i in totdoc:
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    # add tokens to list
    texts.append(stemmed_tokens)
corpus = []
for words in texts:
    corword = ' '.join(words)
    corword = re.sub("[^a-zA-Z 0-9]+", "", corword)
    corpus.append(corword)
print(corpus[:1])

# Creating the Bag of Words model
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print("Bag of Words-->\n", X)
#Creating the Term Document Matrix
count_tokens=cv.get_feature_names()
df_term_doc_vec=pd.DataFrame(data=X,columns=count_tokens)
print("\n Term Document Matrix-->\n",df_term_doc_vec)
# Creating the TF-IDF model
cv2 = TfidfVectorizer()
X2 = cv2.fit_transform(corpus).toarray()
print("\n TF-IDF Matrix-->\n",X2)
# LSA model
print("########### LSA ##############")
# Define the number of topics or components
num_components = 100
# Create SVD object
lsa = TruncatedSVD(n_components=num_components, n_iter=100, random_state=42)
# Fit SVD model on data
Y = lsa.fit_transform(X2)
# Get Singular values and Components
Sigma = lsa.singular_values_
V_transpose = lsa.components_.T
# Print the topics with their terms
terms = cv.get_feature_names()
for index, component in enumerate(lsa.components_):
    zipped = zip(terms, component)
    top_terms_key = sorted(zipped, key=lambda t: t[1], reverse=True)[:5]
    top_terms_list = list(dict(top_terms_key).keys())
    print("Topic " + str(index+1) + ": ", top_terms_list)
df_lsa = pd.DataFrame(Y, columns=['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5'],
index=['Document 1', 'Document 2', 'Document 3', 'Document 4', 'Document 5'])
fig, ax = plt.subplots(figsize=(12, 7))
title = "LSA Clustering Heat-Map"
plt.title(title, fontsize=20)
ttl = ax.title
ttl.set_position([0.5, 1.05])
sns.set(font_scale=1.2)
result = sns.heatmap(df_lsa, fmt=".4f", cmap='Greens', annot=True, annot_kws={"size": 12})
result.set_xticklabels(result.get_xmajorticklabels(), fontsize=14)
result.set_yticklabels(result.get_ymajorticklabels(), fontsize=14, rotation=0)
plt.show()
# LDA model
print("########### LDA ##############")
def topics_print(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:" .format(str(topic_idx+1)) )
        print(" ".join([feature_names[n]
                    for n in topic.argsort()[:-no_top_words - 1:-1]]))
lda_fn = LatentDirichletAllocation(n_components=5, max_iter=100, random_state=40)
Y2 = lda_fn.fit_transform(X)
topic_info = topics_print(lda_fn, count_tokens, 25)
print(topic_info)
df_lda = pd.DataFrame(Y2, columns=['Topic 1', 'Topic 2', 'Topic 3', 'Topic 4', 'Topic 5'],
index=['Document 1', 'Document 2', 'Document 3', 'Document 4', 'Document 5'])
fig, ax = plt.subplots(figsize=(12, 7))
title = "LDA Clustering Heat-Map"
plt.title(title, fontsize=20)
ttl = ax.title
ttl.set_position([0.5, 1.05])
sns.set(font_scale=1.2)
result = sns.heatmap(df_lda, fmt=".4f", cmap='YlGnBu', annot=True, annot_kws={"size": 12})
result.set_xticklabels(result.get_xmajorticklabels(), fontsize=14)
result.set_yticklabels(result.get_ymajorticklabels(), fontsize=14, rotation=0)
plt.show()