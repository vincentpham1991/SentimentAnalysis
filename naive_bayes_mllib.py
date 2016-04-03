import os
import sys
import string

#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#import nltk

# Path for spark source folder
os.environ['SPARK_HOME']="/Users/vincentpham/Downloads/spark-1.5.2/"

# Append pyspark  to Python Path
sys.path.append("/Users/vincentpham/Downloads/spark-1.5.2/python/")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.mllib.feature import HashingTF, IDF
    from pyspark.mllib.regression import LabeledPoint
    from pyspark.mllib.classification import NaiveBayes

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

# Initialize SparkContext

sc = SparkContext('local')

def create_labelPoints(rawdata, label):
    tf = HashingTF().transform(
        rawdata.map(lambda doc: doc[1].lower().split(" "), preservesPartitioning=True))
    return(tf.map(lambda x: LabeledPoint(label, x)))



train_neg_path = "/Users/vincentpham/Documents/Classes/Machine_Learning/aclImdb/train/neg/*.txt"
train_pos_path = "/Users/vincentpham/Documents/Classes/Machine_Learning/aclImdb/train/pos/*.txt"


test_neg_path = "/Users/vincentpham/Documents/Classes/Machine_Learning/aclImdb/test/neg/*.txt"
test_pos_path = "/Users/vincentpham/Documents/Classes/Machine_Learning/aclImdb/test/pos/*.txt"

train_neg_rawdata = sc.wholeTextFiles(train_neg_path)
training_neg = create_labelPoints(train_neg_rawdata, 0)
training_neg.count()

train_pos_rawdata = sc.wholeTextFiles(train_pos_path)
training_pos = create_labelPoints(train_pos_rawdata, 1)

training = training_neg.union(training_pos)




test_neg_rawdata = sc.wholeTextFiles(test_neg_path)
testing_neg = create_labelPoints(test_neg_rawdata, 0)


test_pos_rawdata = sc.wholeTextFiles(test_pos_path)
testing_pos = create_labelPoints(test_pos_rawdata, 1)

testing = testing_neg.union(testing_pos)



# Train a naive Bayes model.
model = NaiveBayes.train(training)

predictionAndLabel = testing.map(lambda p : (model.predict(p.features), p.label))
accuracy = 1.0 * predictionAndLabel.filter(lambda (x, v): x == v).count() / testing.count()
print "the accuracy was", accuracy
#the accuracy was 0.8252
