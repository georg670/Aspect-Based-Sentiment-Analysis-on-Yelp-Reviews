import re # needed to remove special character
from pyspark import Row

import json
from pyspark.sql import SQLContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.mllib.clustering import LDA
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType
from pyspark.sql.functions import *
from nltk.stem.porter import *
from pyspark.sql.types import *

mydir = ('file:/vagrant/project/file/review.json')
sqlContext = SQLContext(sc)
df1 = sqlContext.read.json(mydir)

mydir2 = ('file:/vagrant/project/file/business.json')
df2 = sqlContext.read.json(mydir2)

## joining business and review jsons
df = df1.join(df2,(df1.business_id==df2.business_id)).drop(df2.business_id)

## filtering all the chinese cuisine places from the dataset. 
## You can similarly filter out different cuisines

asian=combined_df.where(
    array_contains(combined_df.categories,"Chinese")|\
    array_contains(combined_df.categories,"Cantonese")|\ 
    array_contains(combined_df.categories, "Taiwanese")|\
    array_contains(combined_df.categories, "Szechuan")) 

## creating a temp table

asian.registerTempTable('asian')
asiandf = sqlContext.sql('select business_id, text from asian')

## storing in a rdd
asianrdd = asiandf.rdd

## keeping only words. No numbers or spaces.

pattern1 = re.compile('\W+|\W+$|[^\w\s]+|_')
pattern2 = re.compile(r'\W*\b\w{1,2}\b')

rdd = asianrdd \
    .mapValues(lambda x: pattern1.sub(' ', x)) \
    .mapValues(lambda x: pattern2.sub(' ', x))

df = rdd.toDF(schema=['file', 'text'])


## creating a dictionary so that each review/text has a unique index to it

row_with_index = Row(*["id"] + df.columns)

def make_row(columns):
    def _make_row(row, uid):
        row_dict = row.asDict()
        return row_with_index(*[uid] + [row_dict.get(c) for c in columns])

    return _make_row

f = make_row(df.columns)

indexed = (df.rdd
           .zipWithUniqueId()
           .map(lambda x: f(*x))
           .toDF(StructType([StructField("id", LongType(), False)] + df.schema.fields)))
           
## tokenizing the reviews, removing stopwords, stemming and storing the results in a dataframe

# tokenize
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
tokenized = tokenizer.transform(indexed)
print 'done'

# remove stop words
stopwordList = ['','get','got','also','really','would','one','good','like','great','tri','love','two','three','took','awesome',
 'me','bad','horrible','disgusting','terrible','fabulous','amazing','terrific','worst','best','fine','excellent','acceptable',
 'my','exceptional','satisfactory','satisfying','super','awful','atrocious','unacceptable','poor','sad','gross','authentic',
 'myself','cheap','expensive','we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
 'weren', 'won', 'wouldn']
 
remover=StopWordsRemover(inputCol="tokens", outputCol="words" ,stopWords=stopwordList)
#remover = StopWordsRemover(inputCol="tokens", outputCol="words",stopword)
cleaned = remover.transform(tokenized)
print 'done'

#stem words
# Instantiate stemmer object
stemmer = PorterStemmer()

# Create stemmer python function
def stem(in_vec):
    out_vec = []
    for t in in_vec:
        t_stem = stemmer.stem(t)
        if len(t_stem) > 2:
            out_vec.append(t_stem)       
    return out_vec

# Create user defined function for stemming with return type Array<String>
stemmer_udf = udf(lambda x: stem(x), ArrayType(StringType()))

# Create new df with vectors containing the stemmed tokens 
# Create new df with vectors containing the stemmed tokens 
vector_stemmed_df = (
    cleaned
        .withColumn("vector_stemmed", stemmer_udf("words"))
  )


# vectorize
cv = CountVectorizer(inputCol="vector_stemmed", outputCol="vectors")
print 'done'
count_vectorizer_model = cv.fit(vector_stemmed_df)
print 'done'
result = count_vectorizer_model.transform(vector_stemmed_df)

corpus = result.select(F.col('id').cast("long"), 'vectors').rdd \
    .map(lambda x: [x[0], x[1]])

# Runnign LDA after processing the data
lda_model = LDA.train(rdd=corpus, k=5, seed=12, maxIterations=50)
# extracting topics
topics = lda_model.describeTopics(maxTermsPerTopic=10)
# extraction vocabulary
vocabulary = count_vectorizer_model.vocabulary

for topic in range(len(topics)):
    print("topic {} : ".format(topic))
    words = topics[topic][0]
    scores = topics[topic][1]
    for word in range(len(words)):
        print(vocabulary[words[word]], "->", scores[word])
