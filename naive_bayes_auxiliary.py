import re
import collections
import numpy as np
import argparse

def words(d):
    """ gets list of word
    :param
        d: texts of the file
    """
    regex = re.compile(r"[^[A-Za-z]")
    words_only = regex.sub(' ', d)
    words_only = words_only.lower()
    wordlist = words_only.split(" ")
    wordlist = [x for x in wordlist if len(x) >= 3]
    return wordlist

def get_text(filename):
    """ reads in file
    :param
        filename: name of file
    """
    with open(filename, 'r') as f:
        text_data = f.read()
    return text_data

def count_words(files, is_bernoulli = False):
    """ map counts to words for a class
    :param
        files: list of files in a specific category
        is_bernoulli: True if calculations are Bernoulli based
    """
    word_map = collections.Counter()
    for f in files:
        text = get_text(f)
        wordlist = words(text)
        if is_bernoulli:
            wordlist = set(wordlist)
        word_map.update(wordlist)
    return word_map

def calculate_prob(word_map, denominator, is_bernoulli = False):
    """ calculate the prob of a word for a training class
    :param
        word_map: counter with count of words
        denominator: what count should be divided by
        is_bernoulli: true if calculations are Bernoulli based
    """
    freq_map = collections.Counter()
    for keys in word_map:
        if is_bernoulli:
            freq_map[keys] = word_map[keys] / float(denominator + 2)
        else:
            freq_map[keys] = (word_map[keys] + 1) / float(denominator)
    return freq_map

def calculate_pclass(wordlist, word_map, denom, prior):
    """ calculates P(c|d)
    :param
        wordlist: list of words in d
        word_map: probability of d(i)|c
        denom: count(c) + |V| + 1
        prior: probability of class c
    """
    prob = np.log(prior)
    words_count = collections.Counter(wordlist)
    for word, count in words_count.items():
        if word in word_map:
            prob += count * np.log(word_map[word])
        else:
            prob += count * np.log(1.0 / denom)
    return prob

def classify(testing, word_map1, word_map2, denom1, denom2, p1, p2,
             is_subjective = False):
    """ Obtain the number of correct class identified
    :param
        testing: documents in testing group
        word_map1: word frequency map of the correct class
        word_map2: word frequency map of the opposite class
        denom1: denominator for the correct class
        denom2: denominator for the opposite class
        p1: prior probability of the correct class
        p2: prior probability of the opposite class
        is_subjective: True if based on filtering subjective sentences
    """
    correct_class = 0
    for t in testing:
        if is_subjective:
            wordlist = testing[t]
        else:
            text = get_text(t)
            wordlist = words(text)
        p_class1 = calculate_pclass(wordlist, word_map1, denom1, p1)
        p_class2 = calculate_pclass(wordlist, word_map2, denom2, p2)
        if p_class1 > p_class2:
            correct_class += 1
    return correct_class

def parseArgument():
    """
    Code for parsing arguments
    """
    parser = argparse.ArgumentParser(description='Parsing a file.')
    parser.add_argument('-d', nargs=1, required=True)
    args = vars(parser.parse_args())
    return args

