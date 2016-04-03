import glob
import random
from datetime import datetime
from naive_bayes_auxiliary import *

def naive_bayes(directory):
    """ Prints the accuracy of the algorithm
    :param
        directory: folder name that contains neg and pos folder.
    """
    neg_files = glob.glob(directory + "/neg/*.txt")
    pos_files = glob.glob(directory + "/pos/*.txt")

    random.shuffle(neg_files)
    random.shuffle(pos_files)

    kfolds = 3
    pos_splits = np.array_split(pos_files, kfolds)
    neg_splits = np.array_split(neg_files, kfolds)

    ave_acc = 0
    for i in range(kfolds):
        pos_tests = pos_splits[i]
        neg_tests = neg_splits[i]
        pos_trains = [x for x in pos_files if x not in pos_tests]
        neg_trains = [x for x in neg_files if x not in neg_tests]

        pos_words = count_words(pos_trains)
        neg_words = count_words(neg_trains)

        len_pos = sum(pos_words.values())
        len_neg = sum(neg_words.values())
        len_v = len(set(pos_words.keys()) | set(neg_words.keys()))

        pos_denom = len_pos + len_v + 1
        neg_denom = len_neg + len_v + 1
        p_given_pos = calculate_prob(pos_words, pos_denom)
        p_given_neg = calculate_prob(neg_words, neg_denom)

        pos_testing_n = len(pos_tests)
        pos_training_n = len(pos_trains)
        neg_testing_n = len(neg_tests)
        neg_training_n = len(neg_trains)

        p_pos = float(pos_training_n) / (pos_training_n + neg_training_n)
        p_neg = float(neg_training_n) / (pos_training_n + neg_training_n)

        pos_correct = classify(pos_tests, p_given_pos, p_given_neg,
                               pos_denom, neg_denom, p_pos, p_neg)
        neg_correct = classify(neg_tests, p_given_neg, p_given_pos,
                               neg_denom, pos_denom, p_neg, p_pos)
        num_correct = pos_correct + neg_correct
        acc = float(num_correct) / (pos_testing_n + neg_testing_n) * 100
        ave_acc += acc
        print("Iteration %d" % (i + 1))
        print("num_pos_test_docs: %d" % pos_testing_n)
        print("num_pos_training_docs: %d" % pos_training_n)
        print("num_pos_correct_docs: %d" % pos_correct)
        print("num_neg_test_docs: %d" % neg_testing_n)
        print("num_neg_training_docs: %d" % neg_training_n)
        print("num_neg_correct_docs: %d" % neg_correct)
        print("accuracy: %1.2f%% \n" % acc)
    ave_acc = ave_acc / kfolds
    print("ave_accuracy: %1.2f%%" % ave_acc)
    return

def main():
    args = parseArgument()
    directory = args['d'][0]
    naive_bayes(directory)
    return

if __name__ == '__main__':
    start = datetime.now()
    main()
    print "\ntime"
    print datetime.now() - start