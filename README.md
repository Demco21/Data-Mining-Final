# Data-Mining-Final
Collect large sum of tweets using a Twitter rest API and perform common data mining techniques on this data
Data Mining Final Program - How it works
Retrieve tweets from Twitter via rest API and assign them a class label using a Support Vector Machine

Modules used: sklearn, numpy, pylab, tweepy

training_set.txt is a pre made training set to use with the Support Vector Machine. It contains tweets
mostly related to automobiles or what you would expect to find in car queries.

alb.txt, mia.txt, nyc.txt, sf.txt are files containing unlabeled queried tweets from each city

albpos.txt, miapos.txt, nycpos.txt, sfpos.txt are files containing only the positive tweets from
the previous files mentions. (these are obtained from the output of svmModel() function)

albfreq.txt, miafreq.txt, nycfreq.txt, sffreq.txt are files containing the most frequent keywords
from the previous mentioned positive files

tweets.txt is a file of tweets with an incrementing count, used for clusters() function
this file is created inside the function as a product of the paramater passed into clusters()

randomtweets.txt are a file of random tweets and labeled_randometweets.txt is the same file but these
tweets are now labeled. labeled_randomtweets is used for calculate_metrics() function.

To collect tweets, run carQuery(num) where num is the number of tweets you want to collect per car, per
city. The tweets will be written to four files, each corresponding to one of Albany, New York City, San
Francisco, and Miami.

To calculate metrics, you use: calculate_metrics(queried_file, unqueried_file)
where the queried_file is a file of LABELED tweets obtained from our carQuery() these tweets
are easily labeled by runing score_tweets(infile) where infile is a file of unlabeled tweets.
unqueried_file is a file of LABELED random tweets obtained from randomQuery(num, outfile) which works similar to 
carQuery(num) except it will obtain random unqueried tweets and write them to outfile.

Using the SVM you can predict the class label of unlabeled tweets by running the SVM like so:
svmModel("training_set.txt", "alb.txt", "output.txt")
Where alb.txt is the testing set and can be swapped for any of the other unlabeled tweet files. This
function will label your tweets using the training set as a reference and write the positive ones to the
output file. A class label of 1 is positive, while a class label of 0 is negative.

You can obtain a file with a list of the most frequent keywords from a file of tweets by running
getFrequency(filenameL, output) where filenameL is a file of all the positive tweets obtained from the output
of the svmModel() function and output is the file you wish to store the list of frequencies.

You can also obtain 5 files of clusters of tweets using the clusters(file) function where file is a file
obtained from the output of svmModel() function.
