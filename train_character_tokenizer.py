import string
import unidecode
from random import sample
import pandas as pd
from char_based_tokenizer import CharacterTokenizer


def preprocessor(sentence):
    chars = 'abcdefghijklmnopqrstuvwxyz' + 'çáãâéêíîóõôú' + '0123456789'
    sentence = sentence.lower()
    sentence_preprocessed = ''
    for word in sentence:
        if word in chars:
            sentence_preprocessed += word
        else:
            sentence_preprocessed += unidecode.unidecode(word)
    return sentence_preprocessed


if __name__ == '__main__':

    chars = 'abcdefghijklmnopqrstuvwxyz' + 'çáãâéêíîóõôú' + '0123456789'
    model_max_length = 100
    tokenizer = CharacterTokenizer(chars, model_max_length)

    dataset = pd.read_csv("./resources/nfce.csv")
    dataset = dataset[dataset["PROD_XPROD"].notna()]

    corpus = dataset["PROD_XPROD"].to_list()

    for desc in corpus:
        print(desc)
        desc = preprocessor(desc)
        print(desc)
        tokens = tokenizer(desc)
        print(tokenizer.decode(tokens["input_ids"]))

    tokenizer.save_pretrained("./tokenizer")
