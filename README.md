Files that must be downloaded from
https://www.cs.cornell.edu/people/pabo/movie-review-data/

"polarity dataset v2.0" -> get the "neg" and "pos" folder
"subjectivity dataset v1.0" -> get "plot.tok.gt9.5000" and  "quote.tok.gt9.5000"
	append .txt at the end of the name

put all of these files in a single folder
	-> in this txt, we'll refer to the generic folder as my_directory
	-> my_directory should contain:
		neg
		pos
		plot.tok.gt9.5000.txt
		quote.tok.gt9.5000.txt


This folder contains three main files:
naive-bayes.py
subjectivity_nb.py
bernoulli_nb.py

And 3 auxiliary files:
naive_bayes_auxiliary.py
subjectivity_auxiliary.py
bernoulli_auxiliary.py

And the files directory:
(my_directory)

How to run:

naive-bayes.py contains the base algorithm.
To run type:
	python naive-bayes -d my_directory


subjectivity_nb.py contains sentences filtered by subjectivity then the base
	classifier is ran.
To run type:
	python subjectivity_nb -d my_directory

Note: you will need the txt files mentioned above in the same folder as
	pos and neg.
Note: algorithm inspired by
	http://www.cs.cornell.edu/home/llee/papers/cutsent.pdf


bernoulli_nb.py contains the algorithm that depends on whether a word was
	present or not.
To run type:
	python bernoulli_nb -d my_directory

Note: algorithm was obtained at
	http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html#fig:bernoullialg
Note: takes longer than the other 2 previous algorithm to run
	(~ 3 minutes on my computer)

