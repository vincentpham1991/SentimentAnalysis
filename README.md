## Sentiment Analysis of Movie Reviews

Files were obtained from [here](https://www.cs.cornell.edu/people/pabo/movie-review-data/). The main task of this machine learning project is to identify whether a movie review is positive or negative. Different variations were used and compared. In addition, I also coded this up in spark using both the mllib library version and a self Spark coded version for naive bayes. 

The specific files that are required to be downloaded are: 

"polarity dataset v2.0" -> get "neg" and "pos" folder

"subjectivity dataset v1.0" -> get "plot.tok.gt9.5000" and  "quote.tok.gt9.5000" (append .txt at the end of the name)

### Files:

###### naive-bayes.py
###### subjectivity_nb.py
###### bernoulli_nb.py
###### naive_bayes_mllib.py
###### naive_bayes_spark.py 



#### And 3 auxiliary files:

###### naive_bayes_auxiliary.py
###### subjectivity_auxiliary.py
###### bernoulli_auxiliary.py

### How to run:

__naive-bayes.py__ contains the base algorithm.

To run type:

	python naive-bayes -d my_directory

where my_directory is the files directory

__subjectivity_nb.py__ contains sentences filtered by subjectivity. Algorithm inspired by [Pang and Lee 2004](http://www.cs.cornell.edu/home/llee/papers/cutsent.pdf) 

To run type:

	python subjectivity_nb -d my_directory

__bernoulli_nb.py__ contains the algorithm that depends on whether a word was present or not. Algorithim described [here](http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html#fig:bernoullialg)
	
To run type:

	python bernoulli_nb -d my_directory

Note: takes longer than the other 2 previous algorithm to run (~ 3 minutes on my computer)

__naive_bayes_mllib.py__ used naive bayes algorithm in Spark's mllib library.

To run, copy and paste code into pySpark shell. 

__naive_bayes_spark.py__ is a self coded a naive bayes algorithm in Spark. 

To run, created an EMR instance and run code on AWS. Alternatively, run it on your local computer but leave out the remote machines part. 
