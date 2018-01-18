import numpy as np
import collections
from keras.callbacks import Callback
WORD_SIZE = 7
from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras import *
import keras
SIZE = 33
const_size_of_matrix = WORD_SIZE*SIZE
ALPHABET = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż "

def count_word(word):
    array_word = list()
    coll = collections.Counter(word)
    for letter in ALPHABET:
        array_word.append(coll[letter])
    return  np.array(array_word)


def vectorise(word):
    word = word.lower()
    if len(word) < WORD_SIZE:
        word = word + " "*(WORD_SIZE-len(word))
    letter_indices = [ALPHABET.find(c)- ALPHABET.find("a") for c in word]
    word_len = len(word)
    vector = []
    for i in letter_indices:
        letter = [0] * SIZE
        letter[i] = 1
        vector += letter
    #Wyrownanie macierzy do rozmiaru 224 - dlugosc maxymalna dla slowa
    for i in range(WORD_SIZE - word_len):
        letter = [0]*SIZE
        vector+=letter
    n = SIZE
    a = []
    for j in range (0,WORD_SIZE):
        a.append(vector[n * j:n * (j + 1)])
    return np.array(a)
def devectorise_v5(word_vector,length_input_word):
    word = ''

    for i in range (0,length_input_word):
        letterIdx = np.argmax(word_vector[i])
        word+= ALPHABET[letterIdx]
    return  word


def generate_array(file_name):

    file = open(file_name, encoding='utf-8',mode= "r")
    words = file.readlines()

    i = 0
    for w in words:
        a = w.strip()
        if len(a)<WORD_SIZE:
            a = a + " "*(WORD_SIZE-len(a))
        words[i] = a
        i += 1

    vectors = [count_word(w) for w in words]

    return np.asarray(vectors)

def generate_array7(file_name):

    file = open(file_name, encoding='utf-8',mode= "r")
    words = file.readlines()

    words = [w.strip() for w in words]

    vectors = []

    for i in range (0, WORD_SIZE):
        vectors.append(list())

    for w in words:
        v = vectorise(w)
        for i in range(0,WORD_SIZE):
            vectors[i].append(v[i])

    return np.asarray(vectors)



def save_results(epochs,batch_size,dense_layer,dense_signle_size,words_train,words_test,accuracy):
    results = open("results.txt", mode="a", encoding="utf-8")
    results.write("\n\n")
    tuple = (epochs,batch_size,dense_layer,dense_signle_size,words_train,words_test,accuracy)
    results.write('\t'.join(str(v) for v in tuple))



class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        counter = 0
        x, y = self.test_data
        predictions = self.model.predict(x)
        predictions = np.asarray(predictions[:int((len(predictions)+1)*.40)])
        counter_lines = 0


        for rows in predictions:
            predicted_word = devectorise_v4(rows)
            if predicted_word != '':
                if predicted_word == words[counter_lines]:
                    counter = counter+1
                    print(predicted_word)
                counter_lines= counter_lines+1
        print('Testing acc:{}%\n'.format((counter/int((len(predictions)+1)*.40))*100))


train_data = generate_array("words_train.txt")
train_data_encoded = generate_array7("words_train.txt")

trainIn = np.asarray(train_data[:])
train_outs = []
for i in range (0,WORD_SIZE):
    train_outs.append(np.asarray(train_data_encoded[i][:]))

n_features = SIZE
n_single_dense_size =512
n_batch_size = 128
n_epochs = 40
##Nalezy recznie policzxyc ile daliscie densow xD
n_dense_layer = 4

main_input = Input(shape=(SIZE, ))

x = Dense(n_single_dense_size, activation='relu')(main_input)
x = Dense(n_single_dense_size, activation='relu')(x)
#x = Dropout(0.2)(x)
x = Dense(n_single_dense_size, activation='relu')(x)
#x = Dropout(0.1)(x)
x = Dense(n_single_dense_size, activation='relu')(x)
x = Dropout(0.2)(x)

output1 = Dense(SIZE, activation='softmax', name='aux_output1')(x)
output2 = Dense(SIZE, activation='softmax', name='aux_output2')(x)
output3 = Dense(SIZE, activation='softmax', name='aux_output3')(x)
output4 = Dense(SIZE, activation='softmax', name='aux_output4')(x)
output5 = Dense(SIZE, activation='softmax', name='aux_output5')(x)
output6 = Dense(SIZE, activation='softmax', name='aux_output6')(x)
output7 = Dense(SIZE, activation='softmax', name='aux_output7')(x)


model = Model(inputs=[main_input], outputs=[output1, output2, output3, output4, output5, output6, output7])
model.compile(optimizer='Nadam', loss='categorical_crossentropy',loss_weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
model.fit([trainIn], [train_outs[0], train_outs[1], train_outs[2], train_outs[3], train_outs[4], train_outs[5], train_outs[6]],
          epochs=n_epochs , batch_size=n_batch_size,verbose=2, validation_split=0.2, shuffle=True,
          #callbacks=[keras.callbacks.EarlyStopping(monitor='aux_output1_loss',
           #           min_delta=0,
            #         patience=25,
             #       verbose=1, mode='auto')],
          )

#model.fit(trainIn,trainOut,batch_size=128, callbacks=[TestCallback((validateIn, validateOut))],epochs=200, verbose=2)
testIn = generate_array("words_test.txt")
predictions = model.predict(testIn)

with open("words_test.txt", encoding='utf-8') as f:
    words = f.readlines()

words = [w.strip() for w in words]

##do przerobienia!!
counter = 0
word_counter =0
list_words = []

prediction_vector = []
for i in range (0,len(testIn)):
    #pobierz elementy z prediction
    for j in range (0,WORD_SIZE):
        prediction_vector.append(predictions[j][i][:])
    predicted_word  = devectorise_v5(prediction_vector, len(words[word_counter]))
    if predicted_word in words :
        counter = counter + 1
        list_words.append(predicted_word)

    word_counter = word_counter+1
    prediction_vector.clear()



n_words_train = len(trainIn)
n_words_test = len(testIn)
n_acc = (len(set(list_words))/int(len(words)))*100

save_results(n_epochs,n_batch_size,n_dense_layer,n_single_dense_size,n_words_train,n_words_test,n_acc)

print('Len of test {}\n'.format(int(len(words))))
print("Correctly found words:{}\n".format(len(set(list_words))))
print("Accuracy:{}%\n".format((len(set(list_words))/int(len(words)))*100))
print(set(list_words))

