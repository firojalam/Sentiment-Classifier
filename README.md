# Sentiment Classifier for Tweet Classification

We designed a sentiment classifier using the dataset discussed in the following section. As a classification algorithms we used CNN and LSTM.

## Data Source
1. sentiment140
2. semeval dataset (2013, 2014, 2015, 2016)

## Prepare semeval data

```
Sentiment_data/Semeval_2017_English_final/GOLD/Subtask_A

cat twitter-2013train-A.txt twitter-2015train-A.txt twitter-2016train-A.txt |awk 'BEGIN{FS="\t"}{print $1"\t"$3"\t"$2}' |grep -v "neutral" >Sentiment_exp/data/semeval_twitter_2013_2015_2016_train.csv

cat twitter-2013dev-A.txt twitter-2016dev-A.txt |awk 'BEGIN{FS="\t"}{print $1"\t"$3"\t"$2}' |grep -v "neutral" >Sentiment_exp/data/semeval_twitter_2013_2015_2016_dev.csv
cat twitter-2013test-A.txt twitter-2014test-A.txt twitter-2015test-A.txt twitter-2016test-A.txt|awk 'BEGIN{FS="\t"}{print $1"\t"$3"\t"$2}' |grep -v "neutral">Sentiment_exp/data/semeval_twitter_2013_2014_2015_2016_test.csv

cd Sentiment_exp/data/

awk 'BEGIN{FS="\t"}{print $NF}' semeval_twitter_2013_2015_2016_train.csv|sort|uniq -c
awk 'BEGIN{FS="\t"}{print $NF}' semeval_twitter_2013_2015_2016_dev.csv|sort|uniq -c
awk 'BEGIN{FS="\t"}{print $NF}' semeval_twitter_2013_2014_2015_2016_test.csv|sort|uniq -c


cd Sentiment_exp/data
cat sentiment140_train.csv >sentiment140_semeval_train.csv
cat semeval_twitter_2013_2015_2016_train.csv >>sentiment140_semeval_train.csv

cat sentiment140_test.csv >sentiment140_semeval_test.csv
cat semeval_twitter_2013_2014_2015_2016_test.csv >>sentiment140_semeval_test.csv

```

## Training the system

THEANO_FLAGS=floatX=float32,device=gpu python bin/cnn_pipeline_feat_semeval.py data/sentiment140_semeval_train_data.csv data/semeval_twitter_2013_2015_2016_dev_data.csv data/sentiment140_semeval_test_data.csv results/sentiment140_semeval_resulst_cnn.txt


## Results on test set using CNN
```
P	R	F1
72.63	69.27	70.18
```

## Results on test set using Bi-LSTM
```
P	R	F1
79.17	75.00	76.54
```


## Classify new data
THEANO_FLAGS=floatX=float32,device=gpu python bin/predict_data.py -c models/sentiment_best.hdf5.config -d data/semeval_twitter_2013_2015_2016_dev_data.csv -o labeled/data/semeval_twitter_2013_2015_2016_dev_labeled.csv
