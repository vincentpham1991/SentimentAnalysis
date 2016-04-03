import numpy as np

AWS_ACCESS_KEY_ID = #Enter Key
AWS_SECRET_ACCESS_KEY = #Enter Secret Key


sc._jsc.hadoopConfiguration().set("fs.s3n.awsAccessKeyId", AWS_ACCESS_KEY_ID)
sc._jsc.hadoopConfiguration().set("fs.s3n.awsSecretAccessKey", AWS_SECRET_ACCESS_KEY)


train_neg_path = "s3n://aclImdb/aclImdb/train/neg/*.txt"
train_pos_path = "s3n://aclImdb/aclImdb/train/pos/*.txt"


test_neg_path = "s3n://aclImdb/aclImdb/test/neg/*.txt"
test_pos_path = "s3n://aclImdb/aclImdb/train/pos/*.txt"

def naive_bayes_train(train_neg_path, train_pos_path):
    ''' Train the Naive Bayes model and return the positive and negative probabilities for each word
    '''
    train_neg_rawdata = sc.wholeTextFiles(train_neg_path)
    train_pos_rawdata = sc.wholeTextFiles(train_pos_path)

    training_neg = train_neg_rawdata.flatMap(lambda x: (x[1].lower().split(" "))).map(lambda x: (x,1)).reduceByKey(lambda x,y: x + y)
    training_pos = train_pos_rawdata.flatMap(lambda x: (x[1].lower().split(" "))).map(lambda x: (x,1)).reduceByKey(lambda x,y: x + y)

    V = (training_neg.union(training_pos)).map(lambda x: x[0]).distinct().count()
    n_neg = training_neg.count()
    n_pos = training_pos.count()

    training_neg_prob = training_neg.map(lambda x: (x[0], np.log((x[1] + 1)/float(V + n_neg + 1))))
    training_pos_prob = training_pos.map(lambda x: (x[0], np.log((x[1] + 1)/float(V + n_pos + 1))))
    return (training_neg_prob, training_pos_prob)

training_neg_prob, training_pos_prob = naive_bayes_train(train_neg_path, train_pos_path)

prior_prob = np.log(.5)

def naive_bayes_classify(test_path, training_neg_prob, training_pos_prob, prior_prob, label):
    ''' given a set of testing files, predict the label.
    '''
    test_rawdata = sc.wholeTextFiles(test_path)

    testing_pos = test_rawdata.flatMapValues(lambda x:  x.lower().split(" "))
    testing_pos = testing_pos.map(lambda x: (x[1], x[0])).join(training_pos_prob)
    testing_pos_reduce = testing_pos.map(lambda x: x[1]).reduceByKey(lambda x,y: x+y +prior_prob)

    testing_neg = test_rawdata.flatMapValues(lambda x:  x.lower().split(" "))
    testing_neg = testing_neg.map(lambda x: (x[1], x[0])).join(training_neg_prob)
    testing_neg_reduce = testing_neg.map(lambda x: x[1]).reduceByKey(lambda x,y: x+y +prior_prob)

    prediction = testing_pos_reduce.join(testing_neg_reduce).map(lambda x: (label, 1 if x[1][0] > x[1][1] else 0))

    return prediction

testing_pos = naive_bayes_classify(test_pos_path, training_neg_prob, training_pos_prob, prior_prob,1)
testing_neg = naive_bayes_classify(test_neg_path, training_neg_prob, training_pos_prob, prior_prob,0)
predictionAndLabel = testing_pos.union(testing_neg)

accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / predictionAndLabel.count()
print "the accuracy was", accuracy

