#Linking the code to my anvil website
!pip install anvil-uplink
import anvil.server
anvil.server.connect("KWLA7V7X3F3F6FMBKN57DO7T-3A62VZNXUKXQYGJA")

import numpy as np
import pandas as pd
import gensim

#Text tokenization utility class
from tensorflow.keras.preprocessing.text import Tokenizer
#Used to pad smaller text files to make them equal to the standard file length to be used, because deep learning model takes only a constant input
from tensorflow.keras.preprocessing.sequence import pad_sequences
#Will be fed the model layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
#Used to test model performance
from sklearn.metrics import classification_report, accuracy_score

#Stored as pandas DataFrame data structure
fakeNews = pd.read_csv('https://github.com/JobandeepKhuman/FakeNewsDetector/releases/tag/v1.0.0/Fake.csv')
realNews = pd.read_csv('https://github.com/JobandeepKhuman/FakeNewsDetector/releases/tag/v1.0.0/True.csv')

#In some records the data that should be in the text column is in the title column and the text column is left empty. To fix this I will combine the
#title and text columns into the text column. The additional title information should also provide the machine learning model with more data to use when
#learning
realNews['text'] = realNews['title'] + " " + realNews['text']
fakeNews['text'] = fakeNews['title'] + " " + fakeNews['text']
#Making all the text lowercase
realNews['text'] = realNews['text'].apply(lambda x: str(x).lower())
fakeNews['text'] = fakeNews['text'].apply(lambda x: str(x).lower())

#Adding a class column to each dataframe so that real and fake news can be differentiated when they are combined
#into the allNews dataFrame
realNews['class'] = 1
fakeNews['class'] = 0

#Removing all columns from except text and class from real and fake news dataframes, as no other columns hold useful information to be processed
realNews = realNews[['text', 'class']]
fakeNews = fakeNews[['text', 'class']]
#Combining the real and fake news dataframes
allNews = realNews.append(fakeNews, ignore_index=True)

#Creating a numpy array storing all the class values
y=allNews['class'].values
#allWords is a list of lists (2D array) Each list represents the text value for a particular news source. Each item in each list
#is each word in that text value
allWords = [d.split() for d in allNews['text'].tolist()]
#Converting the text data to numerical data using the word to vector technique
#sentences = the data that will be used to train the model (must be an iterable of iterables)
#size = the dimensionality of the vectors that will represent the words
#window = the number of nearby words the model will consider (the model learns by seeing what words appear near each other)
#min_count = all words with a total frequency less than min_count are ignored
DIM=100 
w2v_model = gensim.models.Word2Vec(sentences=allWords, size=DIM, window=10, min_count=1)

tokenizer = Tokenizer()
#Updating the internal vocabulary based on a list of texts
tokenizer.fit_on_texts(allWords)
#Mapping each unique word to a unique integer (can be thought of as a unique sequence of digits)
allWords = tokenizer.texts_to_sequences(allWords)

#If a sequence (news source) has less than 1000 words it is padded, if it has more than 1000 words it is truncated
maxlen = 1000
allWords = pad_sequences(allWords, maxlen=maxlen)

#Number of unique words
vocab_size = len(tokenizer.word_index) + 1
vocab = tokenizer.word_index


#Initialising weight matrix of dimenstion numberOfUniqueWords x DimensionOfVectorRepresnetingEachWord
weight_matrix = np.zeros((vocab_size, DIM))

#Weight Matrix = Array of vectors representing each unique word
for word, i in vocab.items():
   weight_matrix[i] = w2v_model.wv[word]

#SETTING UP THE MODEL
#Object to group a linear stack of layers into a Tensorflow.keras.model which has inference and training features
model = Sequential()
#Embedding() turns positive integers into dense vectors of a fixed size
#vocab_size = dimension of input vector
#Input length = length of input sequences (in this case the input sequence will be the sequenxe of vectors that represent the words in a news article)
#Trainable = False means that the weights matrix is not changed (retrained) by the ML model as word2vector has already done this
model.add(Embedding(vocab_size, output_dim=DIM, weights = [weight_matrix], input_length=maxlen, trainable=False))
#Adding a Long Short-Term Memory Layer
#Units = the dimension of the output space
model.add(LSTM(units=128))
#Adding a densley connected Neural Network layer
#Sigmoid is used over softmax as there are only 2 classes: fake and real
model.add(Dense(1, activation='sigmoid'))
#Configuring the model for training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

#Splitting the input data and corresponding target output data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(allWords,y)
#Training the model
model.fit(X_train, y_train, epochs=100)

y_pred = (model.predict(X_test) >=0.5).astype(int)
accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

#Allowing my anvil server to call this function
@anvil.server.callable
def classifier(text):
  print(text[0])
  print(len(text[0]))
  processedText = tokenizer.texts_to_sequences(text)
  print(len(processedText[0]))
  processedText = pad_sequences(processedText, maxlen=1000)
  print(processedText[0])
  result = (model.predict(processedText) >= 0.5).astype(int)
  print(result[0][0])
  return result

#Keeps the notebook running so that the classifier function can be called indefinitley
anvil.server.wait_forever()
