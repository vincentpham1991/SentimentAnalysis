from bernoulli_auxiliary import *
import glob
from datetime import datetime
import random

def bernoulli_nb(directory):
    """ Prints the accuracy of the Bernoulli algorithm
    :param
        directory: folder name that contains neg and pos folder.
    """
    is_bernoulli = True
    pos_files = glob.glob(directory + "/pos/*.txt")
    neg_files = glob.glob(directory + "/neg/*.txt")

    kfolds = 3
    random.shuffle(neg_files)
    random.shuffle(pos_files)

    pos_splits = np.array_split(pos_files, kfolds)
    neg_splits = np.array_split(neg_files, kfolds)

    ave_acc = 0
    for i in range(kfolds):
        pos_tests = pos_splits[i]
        neg_tests = neg_splits[i]
        pos_trains = [x for x in pos_files if x not in pos_tests]
        neg_trains = [x for x in neg_files if x not in neg_tests]

        pos_words = count_words(pos_trains, is_bernoulli)
        neg_words = count_words(neg_trains, is_bernoulli)

        pos_words = combine_dictionary(pos_words, neg_words)
        neg_words = combine_dictionary(neg_words, pos_words)

        pos_testing_n = len(pos_tests)
        pos_training_n = len(pos_trains)
        neg_testing_n = len(neg_tests)
        neg_training_n = len(neg_trains)

        p_given_pos = calculate_prob(pos_words, pos_training_n, is_bernoulli)
        p_given_neg = calculate_prob(neg_words, neg_training_n, is_bernoulli)

        p_pos = float(pos_training_n) / (pos_training_n + neg_training_n)
        p_neg = float(neg_training_n) / (pos_training_n + neg_training_n)

        pos_correct = classify_bern(pos_tests, p_given_pos, p_given_neg,
                                    p_pos, p_neg)
        neg_correct = classify_bern(neg_tests, p_given_neg, p_given_pos,
                                    p_neg, p_pos)

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
    bernoulli_nb(directory)
    return

if __name__ == '__main__':
    start = datetime.now()
    main()
    print "\ntime"
    print datetime.now() - start

