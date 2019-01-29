from __future__ import print_function
import Utils
from keras.models import load_model
import numpy as np

num_encoder_tokens = len(Utils.dico)
num_decoder_tokens = len(Utils.dico_phoneme)
max_encoder_seq_length = Utils.max_word_lengh
max_decoder_seq_length = Utils.max_phoneme_lengh

input_token_index = Utils.dico
target_token_index = Utils.dico_phoneme

# encoder_model = load_model('../model_save/LSTM_80/encoder.h5')
# decoder_model = load_model('../model_save/LSTM_80/decoder.h5')

encoder_model = load_model('encoder.h5')
decoder_model = load_model('decoder.h5')

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


'''
for seq_index in range(1):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    # n = np.random.randint(0, len(input_texts)-1)
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
'''


def encodeWord(word):
    word = word.lower()
    n = len(word)
    result = np.zeros((1, max_encoder_seq_length, len(input_token_index)), dtype='float32')
    for i in range(0, n):
        result[0][i][input_token_index[word[i]]] = 1
    return result


data_test = 'data_train/shuffled_data_test.txt'


def test():
    score = 0
    with open(data_test, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
        nb_lines = len(lines)
    for line in lines[:nb_lines - 1]:
        input_text, output_text = line.split('\t')
        phoneme = decode_sequence(encodeWord(input_text)).split('\n')[0]
        if phoneme == output_text:
            score = score + 1
            print(output_text + " | " + phoneme)
    score = score / nb_lines
    print(score)

test()