import re
import string
import time
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from keras import Model
from keras import backend as K
from keras.layers import Layer
from keras.layers import Concatenate, Dense, TimeDistributed, LSTM, Embedding, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


    
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not","he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam","mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have","mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as","this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would","there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are","we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are","what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is","where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have","why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have","would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all","y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have"}
    
StopWords = set(stopwords.words('english'))


def load_data(file_path):
    reviews = pd.read_csv(file_path)[:5000]
    null_count = reviews.isnull().sum()
    reviews = reviews.dropna()
    print('dropped null value count: ',null_count)
    reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        'Score','Time'], 1)
    reviews = reviews.reset_index(drop=True)
    review.columns = ['headlines','text']
    print(review.head())
    
    return review
    

    
def transform(text):

    transformed_text = text.lower() #convert lowercase
    transformed_text = re.sub(r'\([^)]*\)', '', transformed_text) #removing punctuations and symbols 
    transformed_text = re.sub('"','', transformed_text) #removing " and replacing with space
    transformed_text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in transformed_text.split(" ")]) #replacing contractions based on above dict
    transformed_text = re.sub(r"'s\b","",transformed_text) #removing apostrophe
    transformed_text = re.sub("[^a-zA-Z]", " ", transformed_text) #removing non-alphabetic characters
    transformed_text = ' '.join([word for word in transformed_text.split() if word not in StopWords]) #removing stopwords 

    return transformed_text

    

def preprocess(data):
    
    text_cleaned = []
    summ_cleaned = []

    #creating dataframes for seperating news-text and headlines
    for text in data['text']:
        text_cleaned.append(preprocess(text))
    for summary in data['headlines']:
        summ_cleaned.append(preprocess(summary))
    preprocess_df = pd.DataFrame()
    preprocess_df['text'] = text_cleaned
    preprocess_df['headline'] = summ_cleaned

    #Replacing empty data with nan values 
    preprocess_df['headline'].replace('', np.nan, inplace=True)
    #Drop nan values
    preprocess_df.dropna(axis=0, inplace=True)

    #Adding START and END tokens 
    preprocess_df['headline'] = preprocess_df['headline'].apply(lambda x: '<START>' + ' '+ x + ' '+ '<END>')

    #Finding max length of news and headlines to decide length of decoding sequence
    max_len_news = max([len(text.split()) for text in preprocess_df['text']])
    max_len_headline = max([len(text.split()) for text in preprocess_df['headline']])
    print(max_len_news, max_len_headline)
    print(preprocess_df.head(5))
    
    return preprocess_df, max_len_news, max_len_headline
    
    
def tokenize(clean_df, max_len_news, max_len_headline):

    X_train, X_test, y_train, y_test = train_test_split(clean_df['text'], clean_df['headline'], test_size=0.2, random_state=0)

    #Keras tokenizer for news text.
    news_tokenizer = Tokenizer()
    news_tokenizer.fit_on_texts(list(X_train))
    x_train_seq = news_tokenizer.texts_to_sequences(X_train)
    x_test_seq = news_tokenizer.texts_to_sequences(X_test)
    x_train_pad = pad_sequences(x_train_seq, maxlen=max_len_news, padding='post') #Post padding short texts with 0s.
    x_test_pad = pad_sequences(x_test_seq, maxlen=max_len_news, padding='post')
    #Vocab size of texts.
    news_vocab = len(news_tokenizer.word_index) + 1

    #Keras Tokenizer for summaries.
    headline_tokenizer = Tokenizer()
    headline_tokenizer.fit_on_texts(list(y_train))
    y_train_seq = headline_tokenizer.texts_to_sequences(y_train)
    y_test_seq = headline_tokenizer.texts_to_sequences(y_test)
    y_train_pad = pad_sequences(y_train_seq, maxlen=max_len_headline, padding='post')
    y_test_pad = pad_sequences(y_test_seq, maxlen=max_len_headline, padding='post')
    #Vocab size of summaries.
    headline_vocab = len(headline_tokenizer.word_index) + 1
    
    return x_train_pad,  x_test_pad, y_train_pad, y_test_pad, news_vocab, headline_vocab
    
    
class AttentionLayer(Layer):


    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.U_a = self.add_weight(name='U_a',
                                   shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                   initializer='uniform',
                                   trainable=True)
        self.V_a = self.add_weight(name='V_a',
                                   shape=tf.TensorShape((input_shape[0][2], 1)),
                                   initializer='uniform',
                                   trainable=True)

        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, verbose=False):
        """
        inputs: [encoder_output_sequence, decoder_output_sequence]
        """
        assert type(inputs) == list
        encoder_out_seq, decoder_out_seq = inputs
        if verbose:
            print('encoder_out_seq>', encoder_out_seq.shape)
            print('decoder_out_seq>', decoder_out_seq.shape)

        def energy_step(inputs, states):
            """ Step function for computing energy for a single decoder state """

            assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
            assert isinstance(states, list) or isinstance(states, tuple), assert_msg

            """ Some parameters required for shaping tensors"""
            en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
            de_hidden = inputs.shape[-1]

            """ Computing S.Wa where S=[s0, s1, ..., si]"""
            # <= batch_size*en_seq_len, latent_dim
            reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
            # <= batch_size*en_seq_len, latent_dim
            W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
            if verbose:
                print('wa.s>',W_a_dot_s.shape)

            """ Computing hj.Ua """
            U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
            if verbose:
                print('Ua.h>',U_a_dot_h.shape)

            """ tanh(S.Wa + hj.Ua) """
            # <= batch_size*en_seq_len, latent_dim
            reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
            if verbose:
                print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

            """ softmax(va.tanh(S.Wa + hj.Ua)) """
            # <= batch_size, en_seq_len
            e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
            # <= batch_size, en_seq_len
            e_i = K.softmax(e_i)

            if verbose:
                print('ei>', e_i.shape)

            return e_i, [e_i]

        def context_step(inputs, states):
            """ Step function for computing ci using ei """
            # <= batch_size, hidden_size
            c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
            if verbose:
                print('ci>', c_i.shape)
            return c_i, [c_i]

        def create_inital_state(inputs, hidden_size):
            # We are not using initial states, but need to pass something to K.rnn funciton
            fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
            fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
            fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
            fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
            return fake_state

        fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
        fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

        """ Computing energy outputs """
        # e_outputs => (batch_size, de_seq_len, en_seq_len)
        last_out, e_outputs, _ = K.rnn(
            energy_step, decoder_out_seq, [fake_state_e],
        )

        """ Computing context vectors """
        last_out, c_outputs, _ = K.rnn(
            context_step, e_outputs, [fake_state_c],
        )

        return c_outputs, e_outputs

    def compute_output_shape(self, input_shape):
        """ Outputs produced by the layer """
        return [
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
            tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
        ]
        

def model_train(x_train_pad,  x_test_pad, y_train_pad, y_test_pad, news_vocab, headline_vocab):
    
    
    embedding_dim = 500 #Size of word embeddings.
    latent_dim = 256 #No. of neurons in LSTM layer.

    encoder_input = Input(shape=(max_len_news, ))
    encoder_emb = Embedding(news_vocab, embedding_dim, trainable=True)(encoder_input) #Embedding Layer

    #Three-stacked LSTM layers for encoder. Return_state returns the activation state vectors, a(t) and c(t), return_sequences return the output of the neurons y(t).
    #With layers stacked one above the other, y(t) of previous layer becomes x(t) of next layer.
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
    y_1, a_1, c_1 = encoder_lstm1(encoder_emb)

    encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
    y_2, a_2, c_2 = encoder_lstm2(y_1)

    encoder_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
    encoder_output, a_enc, c_enc = encoder_lstm3(y_2)

    #Single LSTM layer for decoder followed by Dense softmax layer to predict the next word in summary.
    decoder_input = Input(shape=(None,))
    decoder_emb = Embedding(headline_vocab, embedding_dim, trainable=True)(decoder_input)

    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, dropout=0.3, recurrent_dropout=0.2)
    decoder_output, decoder_fwd, decoder_back = decoder_lstm(decoder_emb, initial_state=[a_enc, c_enc]) #Final output states of encoder last layer are fed into decoder.

    #Attention Layer
    attn_layer = AttentionLayer(name='attention_layer') 
    attn_out, attn_states = attn_layer([encoder_output, decoder_output]) 

    decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_output, attn_out])

    decoder_dense = TimeDistributed(Dense(headline_vocab, activation='softmax'))
    decoder_output = decoder_dense(decoder_concat_input)

    model = Model([encoder_input, decoder_input], decoder_output)
    print(model.summary())
    
    #Training the model with Early Stopping callback on val_loss.
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    # callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    model.fit([x_train_pad,y_train_pad[:,:-1]], y_train_pad.reshape(y_train_pad.shape[0],y_train_pad.shape[1], 1)[:,1:]  ,epochs=150, batch_size=8, validation_data=([x_test_pad,y_test_pad[:,:-1]], y_test_pad.reshape(y_test_pad.shape[0],y_test_pad.shape[1], 1)[:,1:]))
    
    return model

def decoded_sequence(input_seq):
    encoder_out, encoder_a, encoder_c = encoder_model.predict(input_seq) 
    #Single elt matrix used for maintaining dimensions.
    next_input = np.zeros((1,1))
    next_input[0,0] = headline_tokenizer.word_index['start']
    output_seq = ''
    #Stopping condition to terminate loop when one summary is generated.
    stop = False
    while not stop:
        #Output from decoder inference model, with output states of encoder used for initialisation.
        decoded_out, trans_state_a, trans_state_c = decoder_model.predict([next_input] + [encoder_out, encoder_a, encoder_c])
        #Get index of output token from y(t) of decoder.
        output_idx = np.argmax(decoded_out[0, -1, :])
        #print(output_idx)
        #If output index corresponds to END token, summary is terminated without of course adding the END token itself.
        if output_idx == headline_tokenizer.word_index['end']:
          # print("end detected.")
          stop = True

        if output_idx>0 and output_idx != headline_tokenizer.word_index['start'] :
            output_token = headline_tokenizer.index_word[output_idx] #Generate the token from index.
            output_seq = output_seq + ' ' + output_token #Append to summary
            out_length = len(output_seq)

#             if int(out_length) > 10:
#               print("hello")
#               stop = True

        #Pass the current output index as input to next neuron.
        next_input[0,0] = output_idx
        #Continously update the transient state vectors in decoder.
        encoder_a, encoder_c = trans_state_a, trans_state_c
        
    return output_seq   


if __name__ == "__main__":
    
    #Load data, preprocess and train model
    review = load_data(file_path)
    clean_df, max_len_news, max_len_headline = preprocess(review)
    x_train_pad,  x_test_pad, y_train_pad, y_test_pad, news_vocab, headline_vocab = tokenize (clean_df, max_len_news, max_len_headline)
    model = model_train(x_train_pad,  x_test_pad, y_train_pad, y_test_pad, news_vocab, headline_vocab)

    #Initalise state vectors for encoder
    encoder_model = Model(inputs=encoder_input, outputs=[encoder_output, a_enc, c_enc])

    #Initialise state vectors for decoder.
    decoder_initial_state_a = Input(shape=(latent_dim,))
    decoder_initial_state_c = Input(shape=(latent_dim,))
    decoder_hidden_state = Input(shape=(max_len_news, latent_dim))

    #Decoder inference model
    decoder_out, decoder_a, decoder_c = decoder_lstm(decoder_emb, initial_state=[decoder_initial_state_a, decoder_initial_state_c])
    attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state, decoder_out])
    decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_out, attn_out_inf])

    decoder_final = decoder_dense(decoder_inf_concat)
    decoder_model = Model([decoder_input]+[decoder_hidden_state, decoder_initial_state_a, decoder_initial_state_c], [decoder_final]+[decoder_a, decoder_c])

    
    #prediction
    i = 30
    print('News:', X_train.iloc[i])
    print('Actual Headline:', y_train.iloc[i])
    print('Predicted Headline:', decoded_sequence(x_train_pad[i].reshape(1, max_len_news)))
    

    
