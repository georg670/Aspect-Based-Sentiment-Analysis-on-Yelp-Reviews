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

df = df1.join(df2,(df1.business_id==df2.business_id)).drop(df2.business_id)
