from naive_bayes_auxiliary import *

def get_sentences(text):
    '''  obtains the sentences from a document
    :param
        text: unfiltered document
    '''
    sentences = re.split("\.|\?|\!", text)
    return sentences

def merge_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.
    :param
        x: dictionary 1
        y: dictionary 2
    Notes: function obtain from stackoverflow
    '''
    z = x.copy()
    z.update(y)
    return z

def agg_words(maps):
    ''' aggregate word count from a map of files
    :param
    maps: map of files which contains word counts for each file
    '''
    counter = collections.Counter()
    for keys in maps:
        counter.update(maps[keys])
    return counter

def classify_subj(sentences, word_map1, word_map2, denom1, denom2, p1, p2):
    ''' Obtains word counts for only subjective sentences for a given document
    :param
        sentences: list of sentences from a document
        word_map1: word frequency map of the subjective class
        word_map2: word frequency map of the objective class
        denom1: denominator for the subjective class
        denom2: denominator for the objective class
        p1: prior probability of the subjective class
        p2: prior probability of the objective class
    '''
    subj_sentences = collections.Counter()
    for s in sentences:
        wordlist = words(s)
        p_subj = calculate_pclass(wordlist, word_map1, denom1, p1)
        p_obj = calculate_pclass(wordlist, word_map2, denom2, p2)
        if p_subj > p_obj:
            subj_sentences.update(wordlist)
    return subj_sentences

def extract_subjectives(polarity_lists, subj_filename, obj_filename, kfolds):
    ''' Obtains a list of filename with word counts for subjective sentences
    :param
    polarity_lists: list of filenames for polarity of interest
    subj_filename: filename for subjective training set.
    obj_filename: filename for objective training set
    kfolds: number of folds
    '''
    subj_map = count_words(subj_filename)
    obj_map = count_words(obj_filename)

    len_subj = sum(subj_map.values())
    len_obj = sum(obj_map.values())
    len_v = len(set(subj_map.keys()) | set(obj_map.keys()))

    subj_denom = len_subj + len_v + 1
    obj_denom = len_obj + len_v + 1

    p_subj = p_obj = .5 #each have 5000 sentences.

    p_given_subj = calculate_prob(subj_map, subj_denom)
    p_given_obj = calculate_prob(obj_map, obj_denom)

    subjective_polarity_list = []

    for k in range(kfolds):
        filename_map = {}
        for filename in polarity_lists[k]:
            text = get_text(filename)
            sentences = get_sentences(text)
            subj_sentences = classify_subj(sentences, p_given_subj, p_given_obj,
                                           subj_denom, obj_denom,
                                           p_subj, p_obj)
            filename_map[filename] = subj_sentences
        subjective_polarity_list.append(filename_map)
    return subjective_polarity_list

