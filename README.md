# text_summarise_attention
Abstractive text summarization using encoder-decoder model with attention layer

Dataset used: Kaggle - https://www.kaggle.com/snap/amazon-fine-food-reviews

This code consists of Encoder-Decoder based transformer model for predicting abstractive text summary (In this case - summary of food reviews) from given text. Attention layer is used to further enhance the predictions since attention layer avoids attempting to learn a single vector representation for each sentence, instead decoder will know how much attention to be paid to each input it recieves based on attention weights. I have used popular "Bahdanau style Attention" here.



References that helped in implementing this algorithm: 

Jaemin Cho's tutorial - https://github.com/j-min/tf_tutorial_plus/tree/master/RNN_seq2seq/contrib_seq2seq
Xin Pan's and Peter Liu's GitHub page - https://github.com/tensorflow/models/tree/master/textsum
