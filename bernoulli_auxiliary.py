from naive_bayes_auxiliary import *

def combine_dictionary(dict1, dict2):
    """ Add words from dictionary 2 that is not in dictionary 1
    :param
        dict1: dictionary 1
        dict2: dictionary 2
    """
    dict1.update(dict1.keys())
    unique_words = [x for x in dict2 if x not in dict1]
    dict1.update(unique_words)
    return dict1

def calc_bern_pclass(wordlist, word_map1, word_map2, prior1, prior2):
    """ calculates if document belongs to class 1
    :param wordlist: list of words in d
    :param word_map1: word map of class 1
    :param word_map2: word map of class 2
    :param prior1: probability of class 1
    :param prior2: probability of class 2
    :return:
    """
    probs1 = np.log(prior1)
    probs2 = np.log(prior2)
    for word in word_map1:
        if word in wordlist:
            probs1 += np.log(word_map1[word])
            probs2 += np.log(word_map2[word])
        else:
            probs1 += np.log(1 - word_map1[word])
            probs2 += np.log(1 - word_map2[word])
    indicator = 0
    if probs1 > probs2:
        indicator = 1
    return indicator



def classify_bern(testing, word_map1, word_map2, p1, p2):
    """ Obtain the number of correct class identified
    :param
        testing: documents in testing group
        word_map1: word frequency map of the correct class
        word_map2: word frequency map of the opposite class
        p1: prior probability of the correct class
        p2: prior probability of the opposite class
    """
    correct_class = 0
    for t in testing:
        text = get_text(t)
        wordlist = words(text)
        wordlist = collections.Counter(set(wordlist))
        indicator = calc_bern_pclass(wordlist, word_map1, word_map2,
                                              p1, p2)
        correct_class += indicator
    return correct_class
