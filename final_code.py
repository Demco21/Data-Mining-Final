#Christopher Snyder, Christian DeMarco, Jonathan Negron
#ICSI431 Data Mining Final Project
 
import tweepy, sys, json, matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
 
consumer_key = 'uGu4hEOYb2J6mdjWvqn8obtSA'
consumer_secret = 'eQtvSeyYPy43ThrHnymkj3Nfc5fEAK8gI3eV0JRk3wGUzX6Dcu'
access_token_key = '742854514957508612-ouGbnSx2IYanANuuoUDLg30dO1mKIn3'
access_token_secret = 'LgPk2bfdTRXQC7GKfYkdGw2hXrrRMQfcFeqMmvP8WN8Wz'

#consumer_key='PeH7lROp4ihy4QyK87FZg'
#consumer_secret='1BdUkBd9cQK6JcJPll7CkDPbfWEiOyBqqL2KKwT3Og'
#access_token_key='1683902912-j3558MXwXJ3uHIuZw8eRfolbEGrzN1zQO6UThc7'
#access_token_secret='e286LQQTtkPhzmsEMnq679m7seqH4ofTDqeArDEgtXw'
 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token_key, access_token_secret)
myApi = tweepy.API(auth)
 
 
stopwords = ["volkswagen", "honda", "toyota", "nissan","audi", "bmw", "hyundai", "chevrolet", "gmc","mitsubishi", "jaguar", "lincoln", "subaru","ram", "mercedes", 
			"jeep", "acura", "ford","dodge", "land rover", "cadillac", "gmc", "chrysler""a", "about", "above", "above", "across", "after", "afterwards", "again", 
			"against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", 
			"any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", 
			"before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", 
			"co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", 
			"elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", 
			"first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", 
			"hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", 
			"indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", 
			"mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", 
			"nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", 
			"our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", 
			"several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", 
			"still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", 
			"thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", 
			"towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", 
			"where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", 
			"with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the", "RT"]

cityDic = {"alb":"42.6525,-73.7571,10mi", "nyc":"40.7128,74.0059,10mi", "sf":"37.7749,122.4194,10mi", "mia":"25.7617,80.1918,10mi"}

carArray = ["volkswagen", "honda", "toyota", "nissan", "audi",
			"bmw", "hyundai", "chevrolet", "gmc", "mitsubishi",
			"jaguar", "lincoln", "subaru", "mercedes", "jeep",
			"acura", "ford", "dodge", "land rover", "cadillac",
			"chrysler"]

randWords = ["the", "is", "have", "who", "what"
			"when", "where", "why", "how", "there",
			"their", "you", "went", "out", "much",
			"would", "could", "should", "do", "keep",
			"first"]
				
#car query gathers tweets related to cars
def carQuery(num):
	print "Gathering tweets..."
	
	for city in cityDic:
		cityFile = city + ".txt"
		for car in carArray:
			get_tweets(num, car, cityDic[city], cityFile)

			
	print "Tweets have been successfully gathered."

	
def randomQuery(num, outfile):
	print "Gathering tweets..."
	
	for word in randWords:
		get_tweets(num, word, cityDic["nyc"], outfile)

			
	print "Tweets have been successfully gathered."
	
	
#Collect a number of tweets (cnt) based on query
def get_tweets(cnt, query, inGeo, file):

	tweets = myApi.search(q=query, geo=inGeo, count=cnt)
 
	with open(file, "a") as outFile:
		for tweet in tweets:
			if not tweet.retweeted and 'RT @' not in tweet.text:
				tweetText = tweet.text.lower()
				output = [tweetText]
				outFile.write(json.dumps(output) + "\n")
 

#Using the tweets collected in the get_tweets() function and store in tweets.txt, this function will
#print out each tweet and allow the user to score it. 0 for neg, 1 for pos. Writes neg and pos
#tweets to their respective files with score added. Also has error handling to make sure score is either 0 or 1
def score_tweets(infile):
	pos = 0
	neg = 0
	for line in open(infile, "r").readlines():
		tweet = json.loads(line)
		score = 2
		while(score != 1 and score != 0):
			try:
				score = int(raw_input(tweet))
				if(score != 1 and score != 0):
					print "TRY ERROR: Invalid score. Enter 0 or 1"
				
			except:
				print "EXCEPT ERROR: Invalid score. Enter 0 or 1"
       
		out = [score, tweet[0]]
 
		if score == 0:
			neg += 1
			with open("training_set.txt", "a") as outfile:
				outfile.write(json.dumps(out) +"\n")
           
		elif score == 1:
			pos += 1
			with open("training_set.txt", "a") as outfile:
				outfile.write(json.dumps(out)+"\n")
		print "neg: ",neg," pos: ",pos
		print("\n")
 
	print "Tweets successfully scored"


#finds the most frequent keywords inside filenameL and write each word with the frequency to output
def getFrequency(filenameL, output):
	tweets = []
	for line in open(filenameL, "r").readlines():
		tweet = json.loads(line)
		tweets.append(["1", tweet[0].lower().strip()])
 
    # Extract the vocabulary of keywords
	vocab = dict()
	for class_label, text in tweets:
		for term in text.split():
			term = term.lower()
			if len(term) > 2 and term not in stopwords:
				if vocab.has_key(term):
					vocab[term] = vocab[term] + 1
				else:
					vocab[term] = 1
 
    # Remove terms whose frequencies are less than 15
	vocab = {term: freq for term, freq in vocab.items() if freq > 15}
	
	f = open(output, "w")
	
	for word in vocab:
		currWord = str(word) + ","
		freq = str(vocab[word])
		f.write(currWord + freq + '\n')


	
#filenameL: training set
#filenameU: testing set
#finds the frequency of keywords in filenameL and uses those keywords to predict the label of tweets in filenameU and writes the positive tweets to output
def svmModel(filenameL, filenameU, output):
	tweets = []
	for line in open(filenameL, "r").readlines():
		tweet = json.loads(line)
		tweets.append([tweet[0], tweet[1].lower().strip()])
 
    # Extract the vocabulary of keywords
	vocab = dict()
	for class_label, text in tweets:
		for term in text.split():
			term = term.lower()
			if len(term) > 2 and term not in stopwords:
				if vocab.has_key(term):
					vocab[term] = vocab[term] + 1
				else:
					vocab[term] = 1
 
    # Remove terms whose frequencies are less than 15
	vocab = {term: freq for term, freq in vocab.items() if freq > 15}
    # Generate an id starting from 0 for each term in vocab
	vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
	print vocab
 
    # Generate X and y
	X = []
	y = []
	for class_label, text in tweets:
		x = [0] * len(vocab)
		terms = [term for term in text.split() if len(term) > 2]
		for term in terms:
			if vocab.has_key(term):
				x[vocab[term]] += 1
		y.append(class_label)
		X.append(x)
 
    # 10 folder cross validation to estimate the best w and b
	svc = svm.SVC(kernel='linear')
	Cs = range(1, 20)
	clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
	clf.fit(X, y)
	
# predict the class labels of new tweets
	print clf.predict(X)
	tweets = []
	for line in open(filenameU).readlines():
		tweets.append(line)
 
# Generate X for testing tweets
	X = []
	for text in tweets:
		x = [0] * len(vocab)
		terms = [term for term in text.split() if len(term) > 2]
		for term in terms:
			if vocab.has_key(term):
				x[vocab[term]] += 1
		X.append(x)
	y = clf.predict(X)
	
    #write all positive tweets to the output file
	f = open(output, "a")
	for idx in range(0, len(tweets)):
		if(y[idx] == 1):
			print 'Sentiment Class (1 means positive; 0 means negative): ', y[idx]
			print 'TEXT: ', idx, tweets[idx]
			labeledTweet = [y[idx], json.loads(tweets[idx])]
			f.write(labeledTweet)
	
	#count = 0
	#for line in tweets:
	#	text = json.loads(line)
	#	label = y[count]
	#	labeledTweet = [label, text[0]]
	#	f.write(json.dumps(labeledTweet)+'\n')
	#	count = count + 1

#clusters each tweet file from each city
def clusters(file):
	count = 0
	f = open("tweets.txt", "w")
	for line in open(file).readlines():
		text = json.loads(line)
		tweetOut = [count,text[0]+'\n']
		f.write(json.dumps(tweetOut) + '\n')
		count = count + 1
	f.close()
	
	tweets = []
	for line in open("tweets.txt").readlines():
		tweets.append(json.loads(line))

	# Extract the vocabulary of keywords
	vocab = dict()
	for tweet_id, tweet_text in tweets:
		for term in tweet_text.split():
			term = term.lower()
			if len(term) > 2 and term not in stopwords:
				if vocab.has_key(term):
					vocab[term] = vocab[term] + 1
				else:
					vocab[term] = 1

	# Remove terms whose frequencies are less than a threshold (e.g., 15)
	vocab = {term: freq for term, freq in vocab.items() if freq > 20}
	# Generate an id (starting from 0) for each term in vocab
	vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}

	# Generate X
	X = []
	for tweet_id, tweet_text in tweets:
		x = [0] * len(vocab)
		terms = [term for term in tweet_text.split() if len(term) > 2]
		for term in terms:
			if vocab.has_key(term):
				x[vocab[term]] += 1
		X.append(x)

	# K-means clustering
	km = KMeans(n_clusters = 5, n_init = 100) # try 100 different initial centroids
	km.fit(X)

	cluster = []
	cluster_stat = dict()
	
	# Print tweets that belong to each cluster
	for idx, cls in enumerate(km.labels_):
		if cluster_stat.has_key(cls):
			cluster_stat[cls] += 1
		else:
			cluster_stat[cls] = 1
		open('cluster-{0}.txt'.format(cls), 'a').write(json.dumps(tweets[idx]) + '\r\n')

	print 'basic information about the clusters that are generated by the k-means clustering algorithm: \r\n'
	print 'total number of clusters: {0}\r\n'.format(len(cluster_stat))
	for cls, count in cluster_stat.items():
		print 'cluster {0} has {1} tweets'.format(cls, count)
	
	
 #calculates the metrics of our SVM
def calculate_metrics(queried_file, unqueried_file):
	
	A = 0.0
	B = 0.0
	C = 0.0
	M = 0.0
	N = 0.0
	
	queried_tweets = []
	for line in open(queried_file, "r").readlines():
		tweet = json.loads(line)
		queried_tweets.append([tweet[0], tweet[1].lower().strip()])
 
	for class_label, text in queried_tweets:
		if class_label == 1:
			A = A + 1
		M = M + 1
		N = N + 1
 
	unqueried_tweets = []
	for line in open(unqueried_file, "r").readlines():
		tweet = json.loads(line)
		unqueried_tweets.append([tweet[0], tweet[1].lower().strip()])
 
	for class_label, text in unqueried_tweets:
		if class_label == 1 and matches(text) == 1:
			B = B + 1
			N = N + 1
		elif class_label == 0 and matches(text) == 1:
			N = N + 1
		elif class_label == 1 and matches(text) == 0:
			C = C + 1
 
	print "A is:",A
	print "B is:",B
	print "C is:",C
 
	api_recall = M / N
	quality_precision = A / M
	quality_recall = A / (A + B + C)
 
	print "API Recall:",api_recall
	print "Quality Precision:",quality_precision
	print "Quality Recall:",quality_recall

#returns 1 if the tweet contains any word from our query or 0 if not
#used when calculating 'B' in getMetrics
def matches(tweet):
	for car in carArray:
		if car in tweet:
			return 1
	return 0
	
#takes in an input of random tweets from infile and writes all tweets that contain a word from our query to outfile
#the purpose of this function is to find all the tweets that are contained in the 'B' box when calculating metrics
def findMatches(infile, outfile):
	f = open(outfile, "w")
	for line in open(infile, "r").readlines():
		tweet = json.loads(line)
		for car in carArray:
			if car in line:
				f.write(json.dumps(tweet)+'\n')
				break
	f.close()
	print "Finished collecting matches."
 

if __name__ == '__main__':
   
	#carQuery(240)
	#randomQuery(100, "randomtweets.txt")
	#calculate_metrics("training_set.txt", "labeled_randomtweets.txt")
	#svmModel("training_set.txt", "nyc.txt", "nycpos.txt")
	#svmModel("training_set.txt", "alb.txt", "albpos.txt")
	#svmModel("training_set.txt", "mia.txt", "miapos.txt")
	#svmModel("training_set.txt", "sf.txt", "sfpos.txt")
	#svmModel("training_set.txt", "randomtweets.txt", "labeled_randomtweets.txt")
	#clusters("albpos.txt")
	#getFrequency("nycpos.txt", "nycfreq.txt")
	#getFrequency("sfpos.txt", "sffreq.txt")
	#getFrequency("albpos.txt", "albfreq.txt")
	#getFrequency("miapos.txt", "miafreq.txt")