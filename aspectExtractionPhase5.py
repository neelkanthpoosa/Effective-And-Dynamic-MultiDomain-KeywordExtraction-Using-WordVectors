import sys
import pprint
import textstat
from sklearn import cluster
from collections import defaultdict
# !{sys.executable} -m spacy download en_core_web_lg
import en_core_web_lg
nlp = en_core_web_lg.load()
import spacy
from spacy import displacy
import json
#import newreviews.json
import pandas as pd
import numpy as np
import ssl
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import matplotlib.pyplot as plt
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import os
import time
from difflib import SequenceMatcher
from spiders.flipkart_reviews import flipkart_scraper
from spiders.snapdeal_reviews import snapdeal_scraper
from spiders.newamazon_reviews import amazon_scraper
from getsearchresults import getreviews
flp = spacy.load("en_core_web_lg", disable=["parser"])
flp.add_pipe(nlp.create_pipe('sentencizer'))

getreviews("https://www.flipkart.com/lenovo-ideapad-s145-core-i3-8th-gen-8-gb-1-tb-hdd-windows-10-home-81vd-s145-15ikb-u-laptop/p/itmf6a8caf01d055?pid=COMFQ7HXAFHFRCEZ&lid=LSTCOMFQ7HXAFHFRCEZIIKFXT&marketplace=FLIPKART&srno=s_1_12&otracker=search&otracker1=search&fm=SEARCH&iid=06aee7a0-0607-436a-a9c9-9d46e79c8d5d.COMFQ7HXAFHFRCEZ.SEARCH&ppt=sp&ppn=sp&ssid=fmmv2pyw2o0000001591470551396&qH=c8c2500230544c3a")

#Data Cleaning

def clean_data(df):

    pd.options.mode.chained_assignment = None

    print("******Cleaning Started*****")

    print(f'Shape of df before cleaning : {df.shape}')
    #df['review_date'] = pd.to_datetime(df['review_date'])
    df = df[df['comment'].notna()]
    df['comment'] = df['comment'].str.replace("<br />", " ")
    df['comment'] = df['comment'].str.replace("\[?\[.+?\]?\]", " ")
    df['comment'] = df['comment'].str.replace("\/{3,}", " ")
    df['comment'] = df['comment'].str.replace("\&\#.+\&\#\d+?;", " ")
    df['comment'] = df['comment'].str.replace("\d+\&\#\d+?;", " ")
    df['comment'] = df['comment'].str.replace("\&\#\d+?;", " ")

    #facial expressions
    df['comment'] = df['comment'].str.replace("\:\|", "")
    df['comment'] = df['comment'].str.replace("\:\)", "")
    df['comment'] = df['comment'].str.replace("\:\(", "")
    df['comment'] = df['comment'].str.replace("\:\/", "")
    df['comment'] = df['comment'].str.replace("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", "")

    #replace multiple spaces with single space
    df['comment'] = df['comment'].str.replace("\s{2,}", " ")
    #replace multiple fullstops by one
    df['comment'] = df['comment'].str.replace('\.+', ".")

    df['comment'] = df['comment'].str.lower()
    print(f'Shape of df after cleaning : {df.shape}')
    print("******Cleaning Ended*****")


    return(df)

#fetching the scraped review files
data = pd.read_csv("newreviews.csv",usecols=[u'comment'])
data1 = pd.read_csv("flipkartreviews.csv",usecols=[u'comment'])
data2 = pd.read_csv("snapdealreviews.csv",usecols=[u'comment'])


vote = pd.read_csv("newreviews.csv",usecols=[u'votes'])
vote1 = pd.read_csv("flipkartreviews.csv",usecols=[u'votes'])
vote2 = pd.read_csv("snapdealreviews.csv",usecols=[u'votes'])


#********************************************************************************************************************************
#VOTES DATA    and general Preprocessing
amazonVP = pd.DataFrame(vote)
twoVP = pd.DataFrame(vote1)
#Preprocessing votes
for i in range(0,len(amazonVP)):#REMOVES PEOPLE LIKE THIS
    amazonVP['votes'][i]=amazonVP['votes'][i][:-26]
    amazonVP['votes'][i]=amazonVP['votes'][i].replace(',',"")
    if(amazonVP['votes'][i]=='One'):
        amazonVP['votes'][i]=1

    amazonVP['votes'][i] = pd.to_numeric(amazonVP['votes'][i])


amazonDF = pd.DataFrame(data)
flipkartDF = pd.DataFrame(data1)
totalVotesDF=amazonVP.append(twoVP).append(vote2)

#REMOVES READMORE
for i in range(0,len(flipkartDF)):
    flipkartDF['comment'][i]=flipkartDF['comment'][i][:-9]


tf = amazonDF.append(flipkartDF).append(data2)


print(len(tf))



print(len(tf['comment']))

result=[]
count=0


df=clean_data(tf)

reviews_decomp=[]
sid=SentimentIntensityAnalyzer()
prod_pronouns = ['it','this','they','these']
#********************************************************************************************************************************
#READABILITY PHASE
AGG=pd.concat([df, totalVotesDF], axis=1)#MERGE

numberOfSentences=[]
votecount=[]
target_words=[]

#READABILITY
def readability(text,vote):
    d = flp(text)
    sentences=(list(sent.string.split('.?') for sent in d.sents))
    numberOfSentences.append(len(sentences))#SENTENCES CHECKED
    votecount.append(vote)#VOTES CHECKED
    target_words.append(sum(x in {"pros", "cons", "advantage","disadvantage"} for x in nltk.wordpunct_tokenize(text)))
    READ_SCORES=textstat.automated_readability_index(text)

    return pd.Series((READ_SCORES,numberOfSentences))

AGG[['scores','sentences']]=AGG.apply(lambda x: readability(x['comment'], x['votes']), axis=1)#CALL FUNC BY ROW
AGG=AGG.assign(targetWords=target_words[:-1])

totalScore=[]

def normalising_readability(v,s,sen,tw):

    ts=(0.3)*np.average(np.asarray(v)) + (0.3*np.average(np.asarray(s))) + (0.2*np.average(np.asarray(tw))) + (0.2*np.average(np.asarray(sen)))
    totalScore.append(ts)
    pass

AGG.apply(lambda x: normalising_readability(x['votes'], x['scores'], x['sentences'], x['targetWords']), axis=1)
print(len(totalScore))
amin= (min(totalScore))
amax= (max(totalScore))
print(amin," ",amax)
for i, val in enumerate(totalScore):
    totalScore[i] = (val-amin) / (amax-amin)

AGG=AGG.assign(TrustabilityScore=totalScore)


#********************************************************************************************************************************

def apply7_extraction(row,nlp,sid):


    stprmv=''
    word_tokens = word_tokenize(row)

    swords=stopwords.words('english')
    not_stopwords_list=['not']
    final_stopwords_list = set([word for word in swords if word not in not_stopwords_list])
    filtered_sentence = [w for w in word_tokens if not w in final_stopwords_list]

    for w in filtered_sentence:
        if w!='#' or w!='*':
            stprmv+=' '+w
    text=stprmv
    e=''
    d=nlp(text)
#         Lemmatising
    for token in d:
        token=token.lemma_
        e=e+token+' '

    doc=nlp(e)
#print("--- SPACY : Doc loaded ---")

    rule1_pairs = []
    rule2_pairs = []
    rule3_pairs = []
    rule4_pairs = []
    rule5_pairs = []
    rule6_pairs = []
    rule7_pairs = []

    for token in doc:
        A = "999999"
        M = "999999"
        if token.dep_ == "amod" and not token.is_stop:
            M = token.text
            A = token.head.text

            # add adverbial modifier of adjective (e.g. 'most comfortable headphones')
            M_children = token.children
            for child_m in M_children:
                if(child_m.dep_ == "advmod"):
                    M_hash = child_m.text
                    M = M_hash + " " + M
                    break

            # negation in adjective, the "no" keyword is a 'det' of the noun (e.g. no interesting characters)
            A_children = token.head.children
            for child_a in A_children:
                if(child_a.dep_ == "det" and child_a.text == 'no'):
                    neg_prefix = 'not'
                    M = neg_prefix + " " + M
                    break

        if(A != "999999" and M != "999999"):
            if A in prod_pronouns or A=='-PRON-' :
                A = "product"
            dict1 = {"noun" : A, "adj" : M, "rule" : 1, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule1_pairs.append(dict1)

        # print("--- SPACY : Rule 1 Done ---")


        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if((child.dep_ == "dobj" and child.pos_ == "ADJ") and not child.is_stop):
                M = child.text
                #check_spelling(child.text)

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M

        if(A != "999999" and M != "999999"):
            if A in prod_pronouns or A=='-PRON-':
                A = "product"
            dict2 = {"noun" : A, "adj" : M, "rule" : 2, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule2_pairs.append(dict2)




        # print("--- SPACY : Rule 2 Done ---")


        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if(child.dep_ == "acomp" and not child.is_stop):
                M = child.text

            # example - 'this could have been better' -> (this, not better)
            if(child.dep_ == "aux" and child.tag_ == "MD"):
                neg_prefix = "not"
                add_neg_pfx = True

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M
                #check_spelling(child.text)

        if(A != "999999" and M != "999999"):
            if A in prod_pronouns or A=='-PRON-':
                A = "product"
            dict3 = {"noun" : A, "adj" : M, "rule" : 3, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule3_pairs.append(dict3)
            #rule3_pairs.append((A, M, sid.polarity_scores(M)['compound'],3))
    # print("--- SPACY : Rule 3 Done ---")



        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if((child.dep_ == "nsubjpass" or child.dep_ == "nsubj") and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if(child.dep_ == "advmod" and not child.is_stop):
                M = child.text
                M_children = child.children
                for child_m in M_children:
                    if(child_m.dep_ == "advmod"):
                        M_hash = child_m.text
                        M = M_hash + " " + child.text
                        break
                #check_spelling(child.text)

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M

        if(A != "999999" and M != "999999"):
            if A in prod_pronouns or A=='-PRON-':
                A = "product"
            dict4 = {"noun" : A, "adj" : M, "rule" : 4, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule4_pairs.append(dict4)
            #rule4_pairs.append((A, M,sid.polarity_scores(M)['compound'],4)) # )

    # print("--- SPACY : Rule 4 Done ---")




        children = token.children
        A = "999999"
        buf_var = "999999"
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if(child.dep_ == "cop" and not child.is_stop):
                buf_var = child.text
                #check_spelling(child.text)

        if(A != "999999" and buf_var != "999999"):
            if A in prod_pronouns or A=='-PRON-':
                A = "product"
            dict5 = {"noun" : A, "adj" : token.text, "rule" : 5, "polarity" : sid.polarity_scores(token.text)['compound']}
            rule5_pairs.append(dict5)
            #rule5_pairs.append((A, token.text,sid.polarity_scores(token.text)['compound'],5))

    # print("--- SPACY : Rule 5 Done ---")



        children = token.children
        A = "999999"
        M = "999999"
        if(token.pos_ == "INTJ" and not token.is_stop):
            for child in children :
                if(child.dep_ == "nsubj" and not child.is_stop):
                    A = child.text
                    M = token.text
                    # check_spelling(child.text)

        if(A != "999999" and M != "999999"):
            if A in prod_pronouns or A=='-PRON-':
                A = "product"
            dict6 = {"noun" : A, "adj" : M, "rule" : 6, "polarity" : sid.polarity_scores(M)['compound']}
            rule6_pairs.append(dict6)

            #rule6_pairs.append((A, M,sid.polarity_scores(M)['compound'],6))

    # print("--- SPACY : Rule 6 Done ---")


        children = token.children
        A = "999999"
        M = "999999"
        add_neg_pfx = False
        for child in children :
            if(child.dep_ == "nsubj" and not child.is_stop):
                A = child.text
                # check_spelling(child.text)

            if((child.dep_ == "attr") and not child.is_stop):
                M = child.text
                #check_spelling(child.text)

            if(child.dep_ == "neg"):
                neg_prefix = child.text
                add_neg_pfx = True

        if (add_neg_pfx and M != "999999"):
            M = neg_prefix + " " + M
        # replace all instances of "it", "this" and "they" with "product"
        if(A != "999999" and M != "999999"):
            if A in prod_pronouns or A=='-PRON-':
                A = "product"
            dict7 = {"noun" : A, "adj" : M, "rule" : 7, "polarity" : sid.polarity_scores(M)['compound']}
            rule7_pairs.append(dict7)
            #rule7_pairs.append((A, M,sid.polarity_scores(M)['compound'],7))



    #print("--- SPACY : Rules Done ---")


    aspects = []

    aspects = rule1_pairs + rule2_pairs + rule3_pairs +rule4_pairs +rule5_pairs + rule6_pairs + rule7_pairs

    
    
    dic = {"aspect_pairs" : aspects}

    return dic

aspect_list = df['comment'].apply(lambda row: apply7_extraction(row,nlp,sid))

print(len(reviews_decomp))
print(aspect_list)
# a = aspect_extraction(nlp,sid)

    # USE THIS IF YOU WANT TO SEE THE ASPECTS IN A FILE
# with open('aspects.txt', 'w') as f:
#     for item in aspect_list:
#         f.write("%s\n" % item)
# print(type(aspect_list))
aspects_tuples=[]
for review in (aspect_list.items()):
    aspect_pairs = review[1]#['aspect_pairs']
#print(aspect_pairs)
    # #FIX HERE
    for number, pairs in enumerate(aspect_pairs['aspect_pairs']):
        print(pairs)
        aspects_tuples.append(pairs)
#{ k:v for k,v in aspects.items() if k[0] == 'noun' }
#key, val = enumerate(aspects[0])
#unique_aspects = list(set(key))
noun=[]
adj=[]
aspects=[]
dictionaryBag=[]
for i in range(0,len(aspects_tuples)):
    noun.append(aspects_tuples[i]['noun'])
    adj.append(aspects_tuples[i]['adj'])
    aspects.append(aspects_tuples[i]['noun'])
    dictionaryBag.append(aspects_tuples[i]['noun'])
    dictionaryBag.append(aspects_tuples[i]['adj'])
unique_aspects = list(set(aspects))
print("UNIQUE NOUNS",unique_aspects,"\n\n")
print('ADJECTIVES',adj,"\n\n")
print(len(unique_aspects))


# need this mapping later for tagging clusters
aspects_map = defaultdict(int)
for asp in aspects:
    aspects_map[asp] += 1


asp_vectors = []
for aspect in unique_aspects:
    #print(aspect)
    token = nlp(aspect)
    asp_vectors.append(token.vector)

# print("\n\n Aspect Vectors",asp_vectors)

NUM_CLUSTERS = 15
if len(unique_aspects) <= NUM_CLUSTERS:
    print(list(range(len(unique_aspects))))

# print("Running k-means clustering...")
n_clusters = NUM_CLUSTERS
kmeans = cluster.KMeans(n_clusters=n_clusters)
kmeans.fit(asp_vectors)
labels = kmeans.labels_


asp_to_cluster_map = dict(zip(unique_aspects,labels))
cluster_id_to_name_map = defaultdict()
cluster_to_asp_map = defaultdict()
freq_map={}
for i in range(NUM_CLUSTERS):
    cluster_nouns = [k for k,v in asp_to_cluster_map.items() if v == i]
#     print(cluster_nouns)
    freq_map = {k:v for k,v in aspects_map.items() if k in cluster_nouns}
    freq_map = sorted(freq_map.items(), key = lambda x: x[1], reverse = True)
#     print(freq_map)
    cluster_id_to_name_map[i] = freq_map[0][0]
    cluster_to_asp_map[i] = cluster_nouns #see clusters better
# cluster_id_to_name_map = defaultdict()
# clusters = set(asp_to_cluster_map.values())
# for i in clusters:
#     this_cluster_asp = [k for k,v in asp_to_cluster_map.items() if v == i]
#     filt_freq_map = {k:v for k,v in aspects_map.items() if k in this_cluster_asp}
#     filt_freq_map = sorted(filt_freq_map.items(), key = lambda x: x[1], reverse = True)
#     cluster_id_to_name_map[i] = filt_freq_map[0][0]
print(aspects_map)
print(freq_map)
print(asp_to_cluster_map,"\n\n\n")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(cluster_to_asp_map)
print(cluster_id_to_name_map)






#BAG OF ASPECTS
dictionaryBag={}
k=''

for i in range(0,len(aspects_tuples)):
    # if(aspects_tuples[i]['noun']=='-PRON-'):
    #     aspects_tuples[i]['noun']='product'
    v=[]
    k=aspects_tuples[i]['noun']
    v.append(aspects_tuples[i]['adj'])
    #v.append(aspects_tuples[i]['polarity'])
    for j in range(1,len(aspects_tuples)):
        if(k==aspects_tuples[j]['noun']):
            v.append(aspects_tuples[j]['adj'])
    dictionaryBag[k]=v


#PRINT FEATURE'S ADJECTIVES
features=[]
for i in cluster_id_to_name_map:
    print("Feature=",cluster_id_to_name_map[i])
    features.append(cluster_id_to_name_map[i])
    print("\n",dictionaryBag[cluster_id_to_name_map[i]],"\n\n\n")

#polarity tests on Features
for i in cluster_id_to_name_map:
    print("Feature=",cluster_id_to_name_map[i])
    for j in range(0,len(dictionaryBag[cluster_id_to_name_map[i]])):
        print("\n",sid.polarity_scores(dictionaryBag[cluster_id_to_name_map[i]][j]))
    print("\n\n")

#CALUCLATING MEAN OF POLARITIES['compund'] OF EACH ASPECT
finalFeaturePol=[]
for i in cluster_id_to_name_map:
    s=[]
    print("Feature=",cluster_id_to_name_map[i])
    for j in range(0,len(dictionaryBag[cluster_id_to_name_map[i]])):
        s.append(sid.polarity_scores(dictionaryBag[cluster_id_to_name_map[i]][j])['compound'])
    xf=sum(s)/len(s)
    finalFeaturePol.append(xf)

print("\n\nASPECTS:")
for i in range(0,len(features)):
    print(features[i],"\t",finalFeaturePol[i],"\n")
print("Unique Aspects",len(unique_aspects))
print("Total Aspects Extracted",len(aspects_tuples))
print("POLARITIES:",finalFeaturePol)

#****************************************************************************************************************
#READABILITY GRAPH CALULATIONS
avg1=sum(votecount)/len(votecount)
avg2=sum(numberOfSentences)/len(numberOfSentences)
avg3=sum(target_words)/len(target_words)
avg4=sum(totalScore)/len(totalScore)
avg4=sum(totalScore)/len(totalScore)
#STORING INDICES OF scores>MEAN
r_Sentences=[]
r_votes=[]
r_tw=[]
trust=[]
ALL_COMMON_REVIEWS=[]
for i in range(0,len(AGG)):
    if((numberOfSentences[i]>avg2)):
        r_Sentences.append(i)
    if((votecount[i]>avg1)):
        r_votes.append(i)
    if((target_words[i]>avg3)):
        r_tw.append(i)
    if((totalScore[i]>avg4)):
        trust.append(i)
    if((totalScore[i]>avg4) and (target_words[i]>avg3)and(votecount[i]>avg1)and(numberOfSentences[i]>avg2)):
        ALL_COMMON_REVIEWS.append(i)

#*************************************************************************************************************************
#PRINT TOP % AND LEAST % REVIEWS
#YET TO BE FIXED(REINDEXING TO BE DONE, RUN AT YOUR OWN ENV)
#DATAFRAME IS CORRECT
AGG.reset_index(inplace=True,drop=True)
BGG=AGG.sort_values(by ='TrustabilityScore')
BGG.reset_index(drop=True,inplace=True)
print("BOTTOM 5 REVIEWS:")
for i in range(0,5):
    print(BGG['comment'][i],"\n\n")
print("\n\n\n")
print("TOP 5 REVIEWS:\n")
for j in reversed(range(len(BGG)-5,len(BGG))):
    print(BGG['comment'][j],"\n\n\n\n")
#*************************************************************************************************************************

#STORING LENGTH OF REVIEWS WHICH CAN HELP YOU PLOT A GRAPH
qq=len(r_Sentences)
ww=len(r_votes)
ee=len(r_tw)
rr=len(trust)
tt=len(ALL_COMMON_REVIEWS)
print(qq,"\t",ww,"\t",ee,"\t",rr,"\t",tt)
plt.bar(["Long Reviews","Popular Reviews","Organised Reviews","Trustable Reviews"], height = [qq,ww,ee,rr])
plt.show()
posvals = []
negvals = []
for g in range(0,len(features)):
    if finalFeaturePol[g] > 0:
        posvals.append(finalFeaturePol[g])
        negvals.append(0)
    else:
        posvals.append(0)
        negvals.append(finalFeaturePol[g])
fig = plt.figure()
ax = plt.subplot(111)
ax.bar(features, negvals, width=0.6, color='r', linewidth = 0.1)
ax.bar(features, posvals, width=0.6, color='b', linewidth = 0.1)
plt.show()
# print("Long Review:")
# print(AGG.loc[numberOfSentences.index(max(numberOfSentences)),'comment'])
#
# print("Popular Review:")
# print(AGG.loc[votecount.index(max(votecount)),'comment'])
#
# print("Organised Review:")
# print(AGG.loc[target_words.index(max(target_words)),'comment'])
#
# print("Overall Trustable Review:")
# print(AGG.loc[totalScore.index(max(totalScore)),'comment'])

# print('SCORE:',READ_SCORES[1])

#DISPLACY demo
# doc = nlp(df['comment'][1])
# sentence_spans = list(doc.sents)
# displacy.serve(sentence_spans, style="dep")
