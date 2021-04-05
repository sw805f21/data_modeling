import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os

from PIL import Image
import glob

import io
import time
from tensorflow.keras.preprocessing import image_dataset_from_directory



def add_start_end_tokens(num_examples):
    path_to_file_germany = "data\phoenix2014T.train.de"
    # Using readlines()
    file1 = open(path_to_file_germany, 'r')
    Lines = file1.readlines()

    processed_lines = []

    count = 0
    # Strips the newline character
    for line in Lines:
        processed_lines.append("<start> " + line.strip() + " <end>")
        if(len(processed_lines)) == num_examples:
            break 
    return processed_lines

def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",") i.e. removing special characters # german characters rip
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
                 if unicodedata.category(c) != 'Mn')


def preprocess_german_sentence(w):
  w = w.strip()
  w = '<start> ' + w + ' <end>'
  return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(path, num_examples):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n') # strip removes space at beginning and end, split splits file into a list, seperator \n

  word_pairs = [[preprocess_sentence(w) for w in line.split('\t')] # \t is horizontal whitespace, e.g. whitespace. in this case it iterates through each line and splits them by character
                for line in lines[:num_examples]]
  return zip(*word_pairs)


def create_target_dataset(path, num_examples):
  lines = io.open(path).read().split('\n') # strip removes space at beginning and end, split splits file into a list, seperator \n

  preprocessed = [[preprocess_german_sentence(w) for w in line]
                for line in lines[:num_examples]]

  return preprocessed

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = create_dataset(path, num_examples)

  # Changes target language to german
  targ_lang = add_start_end_tokens(num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

# Print what numbers in tensor correspond to which letter
def convert(lang, tensor):
  for t in tensor:
    if t != 0:
      print(f'{t} ----> {lang.index_word[t]}')


class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim) # Turns positive integers (indexes) into dense vectors of fixed size.
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state=hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # Passes the random numbers through an embedding layer, perhaps to have a  layer that learns best representation of numbers for embedding. 
    # That's why it is initalized to random numbers, the numbers are there for initliazation only.
    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # Concacts attention and embedding layer
    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


def build_mobilenet(shape=(224, 224, 3), nbout=3):
    model = keras.applications.mobilenet.MobileNet(
        include_top=False,
        input_shape=shape,
        weights='imagenet')

    classification_head_layers = 30
    for layer in model.layers[:-classification_head_layers]:
        layer.trainable = False

    output = GlobalMaxPool2D()
    return keras.Sequential([model, output])


@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0
  # Operations in a gradient tape is used for training. We are manually setting the context because we are not using variables created by tf.Variable
  # By default, the resources held by a GradientTape are released as soon as GradientTape.gradient() method is called.
  with tf.GradientTape() as tape: 
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    # decoder input is the start token, expand dims to add dimension and create tensor for computation. The shape is (shape of tensor with start token, batch size, 1).
    # first the start token is added to a tensor and this tensor is used to calculate shape afterwards. [targ_lang.word_index['<start>']] * BATCH_SIZE, 1
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1) # Finds id for '<start>' in word index

    # Teacher forcing - feeding the target as the next input
    # for loop from 1 to index [1] in shape(). target is probably a sentence, where you now feed the next word 
    # after '<start>' token as next input and repeats to end of sentence.
    # Makes sentence translation to word t, until whole sentence have been translated
    for t in range(1, targ.shape[1]):  
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      # Checks target sentence with predicted sentence
      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing # new decoder input, so it have previous part of target sentence so it can try to predict the next word, given it have guessed correctly the previous parts
      # of the sentence # teacher forcing, instead of going on tangent of a wrong translation it will start from a correct start point, a teacher "forcing" it on track 
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables) # Tell with the tape, that we want to train both encoder and decoder trainable variables / weights

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')] # split sentence to word list with space as seperator
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post') # pads once again so input have same length, padding is added at end of tensor therefore 'post'
  # think this is necessary because he padded word arrays to be of same length, but have yet to convert the words to a tensor. 
  # can convert python list to tensor
  inputs = tf.convert_to_tensor(inputs) 

  result = ''

  # does same as training 
  hidden = [tf.zeros((1, units))] # initalized tensor that is (1, units)
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  # is max length target longest possible translation of whole text or sentence? probably sentence, given that it returns at <end> token?
  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy() # takes the most likely predicted word id

    result += targ_lang.index_word[predicted_id] + ' ' # converts id to a word and concatenates it to the result

    # if it predicts the most likely id to be a end of sentence token, return the sentence.
    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

  # function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10, 10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input:', sentence)
  print('Predicted translation:', result)

  attention_plot = attention_plot[:len(result.split(' ')),
                                  :len(sentence.split(' '))]
  plot_attention(attention_plot, sentence.split(' '), result.split(' '))

# -------------------------------------- Main Code --------------------------------


_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

en_sentence = u"May I borrow this book?"
sp_sentence = u"¿Puedo tomar prestado este libro?"
print(preprocess_sentence(en_sentence))
print(preprocess_sentence(sp_sentence).encode('utf-8'))

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

# Tests createdataset function
en, sp = create_dataset(path_to_file, None)
sp = add_start_end_tokens(30) 
print(en[-1])
print(sp[-1])

# Try experimenting with the size of that dataset
num_examples = 30# 30 for german

input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file,
                                                                num_examples)

# Calculate max_length of the target tensors
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

# Creating training and validation sets using an 80-20 split, training set is used when training the model, validation set is used after training to see how well
# it performs on data it has not yet seen and calculate a sepearate validation error. (Test set is the completely new dataset for competitions, it have had no chance to change its hyperparameters for)
# There is input and target for both train and validation, because it needs to be fed a input in both cases, and check if it matches the expected target


# Load Image Dataset
train_dir = "dataset_videoes\\train"
path, dirs, files = next(os.walk(train_dir))

np_array_imgs_all_dirs = []

for dir_path in dirs:
    filelist = glob.glob('dataset_videoes\\train\\' + dir_path + '\\*.png')
    x = np.array([np.array(Image.open(fname)) for fname in filelist])
    np_array_imgs_all_dirs.append(x)

np_array_imgs_all_dirs = np.array(np_array_imgs_all_dirs, dtype=object) # Can be removed, depends on what you want the first element of the tuple to be

# TYPES: <class 'numpy.ndarray'>, both input_tensor and input_tensor_train # whats train and val size after this since it splits it but never uses val? waste of data?
#input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(np_array_imgs_all_dirs, target_tensor, test_size=0.2)





print("Input Language; index to word mapping")
#convert(inp_lang, input_tensor_train[0])
print()
print("Target Language; index to word mapping")
convert(targ_lang, target_tensor[0])

BUFFER_SIZE = len(np_array_imgs_all_dirs)
BATCH_SIZE = 1
steps_per_epoch = len(np_array_imgs_all_dirs)//BATCH_SIZE # We want to feed all the data in batches of 64, so we calcualte steps we will have to do this in, in each epoch (training session)
embedding_dim = 256 # how many different words?, why static 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1 # .word seems to be able to called both as list and dic, and in this case just calling it seems to get its size
vocab_tar_size = len(targ_lang.word_index)+1


dataset = tf.data.Dataset.from_generator(lambda: (np_array_imgs_all_dirs, target_tensor), tf.int32).shuffle(BUFFER_SIZE) # Not sure about the tf.int32 or if its necessary to shuffle
#dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE) # slices tensor so each list element of sentences in the

dataset = dataset.batch(BATCH_SIZE, drop_remainder=True) # makes the list of tensors to batches

example_input_batch, example_target_batch = next(iter(dataset)) # dynamically makes dataset an iterator, and takes takes the next element, dataset consist of both input and target train 

#example_input_batch.shape, example_target_batch.shape

