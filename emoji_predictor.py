import numpy as np
import pandas as pd

train = pd.read_csv('train_emoji.csv',header=None)
test = pd.read_csv('test_emoji.csv',header=None)

print(train.head())

import emoji as emoji

#emoji.EMOJI_ALIAS_UNICODE

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":beaming_face_with_smiling_eyes:",
                    "3": ":downcast_face_with_sweat:",
                    "4": ":fork_and_knife:",
                   }

emoji.emojize(":fork_and_knife:")

for e in emoji_dictionary.values():
    print(emoji.emojize(e))

data = train.values
for i in range(10):
    print(data[i][0],emoji.emojize(emoji_dictionary[str(data[i][1])]))

from keras.utils import to_categorical

XT = train[0]
Xt = test[0]

YT = to_categorical(train[1])
Yt = to_categorical(test[1])

print(XT.shape)
print(Xt.shape)
print(YT.shape)
print(Yt.shape)

embeddings = {}
with open('glove.6B.50d.txt',encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:],dtype='float32')
        
        #print(word)
        #print(coeffs)
        embeddings[word] = coeffs

def getOutputEmbeddings(X):
    
    embedding_matrix_output = np.zeros((X.shape[0],10,50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]
            
    return embedding_matrix_output

emb_XT = getOutputEmbeddings(XT)
emb_Xt = getOutputEmbeddings(Xt)

print(emb_XT.shape)
print(emb_Xt.shape)

from keras.layers import *
from keras.models import Sequential
model = Sequential()
model.add(LSTM(64,input_shape=(10,50),return_sequences=True))
model.add(Dropout(0.4))
model.add(LSTM(64,input_shape=(10,50)))
model.add(Dropout(0.3))
model.add(Dense(5))
model.add(Activation('softmax'))
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
model.fit(emb_XT,YT,batch_size=32,epochs=40,shuffle=True,validation_split=0.1)

model.evaluate(emb_Xt,Yt)

pred = model.predict_classes(emb_Xt)

for i in range(30):
    print(' '.join(Xt[i]))
    print(emoji.emojize(emoji_dictionary[str(np.argmax(Yt[i]))]))
    print(emoji.emojize(emoji_dictionary[str(pred[i])]))