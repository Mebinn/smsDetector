from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
import numpy as np
import inflect
import re

def sanitizeText(txt):
    string = txt
    string = re.sub('[€]', 'e', string)
    string = re.sub('[@]', 'a', string)
    string = string.replace('bit.ly', ' Hyper link ')
    string = string.replace('http', ' Hyper link ')
    string = string.replace('www', ' Hyper link ')
    string = re.sub('[%]', ' percent ', string)
    string = re.sub('[*]', ' asterisk ', string)
    string = re.sub('[$]', ' dollars ', string)
    string = re.sub('[£]', ' dollars ', string)
    string = re.sub('[=]', ' ', string)
    string = string.replace('/month', ' per month ')

    for word in string.split():
        try:
            var = float(word)
            var = p.number_to_words(var)
            string = re.sub(word, var, string)
        except:
            notafloat = False
    return string

if __name__ == "__main__":
    p = inflect.engine()

    #This portion opens the dataset and saves the label and feature into two lists: oldListOfSentences and oldListOfLabels
    oldListOfSentences = []
    oldListOfLabels = []

    f = open("SMSSpamCollection.txt","r")

    for line in f:
        words=line.split()
        if words:
            spam = words[0]
            list = words[1:]
            sentence = ""
            for ele in list:
                sentence += ele + " "
            sentence = sanitizeText(sentence)

            oldListOfLabels.append(spam)
            oldListOfSentences.append(sentence)
    f.close()
    #End of portion


    #This portion counts the number of spam in the dataset
    no_of_spam = 0
    no_of_ham = 0
    for element in oldListOfLabels:
        if element[0:4] == "spam":
            no_of_spam +=1
        else:
            no_of_ham += 1
    #End of portion


    #This portion reduces the number of ham messages in the list until it is equivalent to the number of spam messages
    #This is done to make sure the accuracy is better
    index = 0
    ham_count = 0
    indices=[]
    for x in oldListOfLabels:
        if x[0:4] != "spam":
            if ham_count >= no_of_spam:
                indices.append(index)
                ham_count+=1
            else:
                ham_count += 1

        index += 1

    listOfLabels = [v for i, v in enumerate(oldListOfLabels) if i not in indices]
    listOfSentences = [v for i, v in enumerate(oldListOfSentences) if i not in indices]
    #End of portion


    #This portion takes 90% of the data for training and 10% for testing
    split = round(len(listOfSentences)*0.9)
    train_sms = listOfSentences[:split]     #Messages for training (90%)
    train_lab = listOfLabels[:split]        #Labels for training (90%)
    test_sms = listOfSentences[split:]      #Messages for testing (10%)
    test_lab = listOfLabels[split:]         #Labels for testing(10%)

    # This is to store the converted train_lab list. For tensorflow to read the data, the labels must be converted to 1 and 0
    train_label = []
    # This is to store the converted test_lab list. For tensorflow to read the data, the labels must be converted to 1 and 0
    test_label = []

    for ele in train_lab:
        if ele[0:4] =="spam":
            train_label.append(1)
        else:
            train_label.append(0)


    for ele2 in test_lab:
        if ele2[0:4] =="spam":
            test_label.append(1)
        else:
            test_label.append(0)
    #End of portion


    #Pre-processing the data for training
    vocab_size = 10000
    embedding_dim = 16
    max_length = 100
    trunc_type = 'post'
    oov_tok = '<OOV>'
    padding_type = 'post'

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok, char_level=False)
    tokenizer.fit_on_texts(train_sms)
    word_index = tokenizer.word_index

    sequences = tokenizer.texts_to_sequences(train_sms)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    testing_sentences = tokenizer.texts_to_sequences(test_sms)
    testing_padded = pad_sequences(testing_sentences, maxlen=max_length,padding = padding_type, truncating = trunc_type)
    #End of portion


    #Preparing the model and training/testing it
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(25, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    train_label_final = np.array(train_label)
    test_label_final = np.array(test_label)

    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
    history = model.fit(padded, train_label_final, epochs=30, validation_data=(testing_padded,test_label_final))

    results = model.evaluate(testing_padded, test_label_final)
    print(results)
    #End of portion


    #Prediction portion

    def predict(txt):
        if(str(type(txt)) == "<class 'str'>"):
            txt = sanitizeText(txt)
            se = tokenizer.texts_to_sequences(txt)
            pa = pad_sequences(se, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            result = model.predict(pa)
            print("works")
            return(result[0])
        else:
            newlist = []
            for x in txt:
                newtxt = sanitizeText(x)
                newlist.append(newtxt)
            se = tokenizer.texts_to_sequences(newlist)
            pa = pad_sequences(se, maxlen=max_length, padding=padding_type, truncating=trunc_type)
            return (model.predict(pa))

    sample = "Lai lai play ML. Show yall who the legend"
    print(predict(sample))

    predict_msg = ["You have WON a guaranteed £1000 cash or a £2000 prize. To claim yr prize call our customer service representative on 08714712412 between 10am-7pm Cost 10p",
     "Our records indicate u maybe entitled to 5000 pounds in compensation for the Accident you had. To claim 4 free reply with CLAIM to this msg. 2 stop txt STOP",
     "Spook up your mob with a Halloween collection of a logo & pic message plus a free eerie tone, txt CARD SPOOK to 8007 zed 08701417012150p per logo/pic "]
    print(predict(predict_msg))

    predict_msg2 = [
         "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
         "Ok lar... Joking wif u oni...",
         "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
    print(predict(predict_msg2))

    predict_msg3 = ["You are awarded a Nikon Digital Camera. Call now",
                    "Call me",
                    "What's up?"]
    print(predict(predict_msg3))

    #End of portion