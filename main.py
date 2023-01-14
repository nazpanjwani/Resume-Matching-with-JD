from itertools import count
import operator
import math
from ast import pattern
from glob import glob
import json
from operator import index
import os
import numpy
from spacy.pipeline import EntityRuler
import spacy
from spacy import displacy
from scipy import spatial
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter import *
import sys
Pstem=PorterStemmer()
i=-1
j=-1
nlp=spacy.load("en_core_web_sm")
V_soft=[]
V_soft_stem=[]
V_tech=[]
V_tech_stem=[]
index_tech={}
index_dup_tech={}
index_soft={}
index_dup_soft={}

def create_index():
    index_dup={}
    DIR = 'cvs txt/'
    sourcepath = os.listdir(DIR)
    for file in sourcepath:
        inputfile = DIR + file
        f1=open(inputfile,encoding="utf8")
        file_con=f1.read().lower()
        file_con=file_con.lower().replace('\t',' ')
        file_con=file_con.replace('\n',' ')
        file_con=file_con.replace('.','')
        file_con=file_con.replace('. '," ")
        file_con=file_con.replace(',',' ')
        file_con=file_con.replace('-',' ')
        file_con=file_con.replace(')',' ')
        file_con=file_con.replace('(',' ')
        file_con=file_con.replace('%',' ')
        file_con=file_con.replace('@',' ')
        file_con=file_con.replace(' +',' ')
        file_con=file_con.replace(':',' ')
        file_con=file_con.replace('/',' ')
        file_con=file_con.replace('_',' ')
        file_con=file_con.replace('--',' ')
        file_con=file_con.replace('[',' ')
        file_con=file_con.replace(']',' ')
        file_con=file_con.replace(' #',' ')
        file_con=file_con.replace('&',' ')
        file_con=file_con.replace('"',' ')
        file_con=file_con.replace('\uf0b7',' ')
        file_con=file_con.replace('–',' ')
        file_con=file_con.replace('“',' ')
        file_con=file_con.replace('”',' ')
        file_con=file_con.replace('•',' ')
        file_con=file_con.replace('\ufeff',' ')
        file_con=file_con.replace('’',' ')
        file_con=file_con.replace('|',' ')
        file_con=file_con.replace('\x01',' ')
        file_con=file_con.replace('\uf0a7',' ')
        
        f = open("Stopword-List.txt", 'r')
        for line in f:
            for st in line.split():
                file_con=file_con.lower().replace(' '+st+' ',' ')
        f.close()
        doc=word_tokenize(file_con)
        index_dup[file]=doc
    return index_dup

def create_query_ind(filename):
    f2=open(filename,'r')
    file_con=f2.read().lower()
    file_con=file_con.lower().replace('\t',' ')
    file_con=file_con.replace('\n',' ')
    file_con=file_con.replace('.','')
    file_con=file_con.replace('. '," ")
    file_con=file_con.replace(',',' ')
    file_con=file_con.replace('-',' ')
    file_con=file_con.replace(')',' ')
    file_con=file_con.replace('(',' ')
    file_con=file_con.replace('%',' ')
    file_con=file_con.replace('@',' ')
    file_con=file_con.replace(' +',' ')
    file_con=file_con.replace(':',' ')
    file_con=file_con.replace('/',' ')
    file_con=file_con.replace('_',' ')
    file_con=file_con.replace('--',' ')
    file_con=file_con.replace('[',' ')
    file_con=file_con.replace(']',' ')
    file_con=file_con.replace(' #',' ')
    file_con=file_con.replace('&',' ')
    file_con=file_con.replace('"',' ')
    file_con=file_con.replace('\uf0b7',' ')
    file_con=file_con.replace('–',' ')
    file_con=file_con.replace('“',' ')
    file_con=file_con.replace('”',' ')
    file_con=file_con.replace('•',' ')
    file_con=file_con.replace('\ufeff',' ')
    file_con=file_con.replace('’',' ')
    file_con=file_con.replace('|',' ')
    file_con=file_con.replace('\x01',' ')
    file_con=file_con.replace('\uf0a7',' ')
    # file_con=file_con.lower().replace('\t',' ').replace('\n',' ').replace('.','').replace('. '," ").replace(',',' ').replace('-',' ').replace(')',' ').replace('(',' ').replace('%',' ').replace('@',' ').replace(' +',' ').replace(':',' ').replace('/',' ').replace('_',' ').replace('--',' ').replace('[',' ').replace(']',' ').replace(' #',' ').replace('&',' ').replace('"',' ').replace('\uf0b7',' ').replace('–',' ').replace('“',' ').replace('”',' ').replace('•',' ').replace('\ufeff',' ').replace('’',' ').replace('|',' ').replace('\x01',' ').replace('\uf0a7',' ')
    f = open("Stopword-List.txt", 'r')
    for line in f:
        for st in line.split():
            file_con=file_con.lower().replace(' '+st+' ',' ')
    f.close()
    doc=word_tokenize(file_con)
    stemming=[]
    for w in doc:
        stemming.append(Pstem.stem(w))
    doc=stemming
    return doc

#Reads in all created patterns from a file and adds it to the pipeline
def add_newruler_to_pipeline(): 
    data=open('skill_patterns.jsonl','r')
    pattern=[]
    for line in data:
        sl=line.strip()
        pattern.append(json.loads(sl))
    new_ruler=nlp.add_pipe("entity_ruler")
    new_ruler.add_patterns(pattern)

#Visualize the Skill entities of a doc
def visualize_entity_ruler(entity_list, doc):    
    options = {"ents": entity_list}
    displacy.render(doc, style='ent', options=options)
    
#Create dictionary for softskills of candidates
def create_softskill_set(resume_names,doc):
    '''Create a set of the extracted skill entities of a doc'''
    global j
    j=j+1
    soft_sk=set([ent.label_.upper()[7:] for ent in doc.ents if 'softsk' in ent.label_.lower()])
    index_soft[resume_names[j]]=list(soft_sk)
    for w in soft_sk:
        V_soft.append(w)
    t1=[]
    for w in soft_sk:
        if w not in t1:
            t1.append(w)
    index_dup_soft[resume_names[j]]=list(t1)
    return soft_sk
         
def create_softskill_set_JD(doc):
    '''Create a set of the extracted skill entities of a doc'''
    return set([ent.label_.upper()[7:] for ent in doc.ents if 'softsk' in ent.label_.lower()])
  
def create_technicalskill_set_JD(doc):
    '''Create a set of the extracted skill entities of a doc'''
    return set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()])

#Create dictionary for technical skills of candidates 
def create_technicalskill_set(resume_names,doc):
    global i
    i=i+1
    '''Create a set of the extracted skill entities of a doc'''
    tech_sk=set([ent.label_.upper()[6:] for ent in doc.ents if 'skill' in ent.label_.lower()])
    index_tech[resume_names[i]]=list(tech_sk)
    for w in tech_sk:
        V_tech.append(w)
    t1=[]
    for w in tech_sk:
        if w not in t1:
            t1.append(w)
    index_dup_tech[resume_names[i]]=list(t1)
    return tech_sk

def create_education_set(doc):
    return set([ent.label_.upper()[10:] for ent in doc.ents if 'education' in ent.label_.lower()])

def create_skillset_dict(resume_names, resume_texts):
    '''Create a dictionary containing a set of the extracted skills. Name is key, matching skillset is value'''
    softskillsets = [create_softskill_set(resume_names,resume_text) for resume_text in resume_texts]
    technicalskillsets = [create_technicalskill_set(resume_names,resume_text) for resume_text in resume_texts]
    educationset = [create_education_set(resume_text) for resume_text in resume_texts]

    return dict(zip(resume_names, softskillsets)), dict(zip(resume_names, technicalskillsets)), dict(zip(resume_names, educationset)) 
  
def create_tokenized_texts_list():
    '''Create two lists, one with the names of the candidate and one with the tokenized 
       resume texts extracted from either a .pdf or .doc'''
    resume_texts, resume_names = [], []
    DIR = 'cvs txt/'
    sourcepath = os.listdir(DIR)
    for file in sourcepath:
        inputfile = DIR + file
        f1=open(inputfile,encoding="utf8")
        fus=f1.read().lower()
        fus=fus.lower().replace('\t',' ').replace('\n',' ').replace('.','').replace('. '," ").replace(', ',' ').replace('-',' ').replace(')',' ').replace('(',' ').replace('%',' ').replace('@',' ').replace(' +',' ').replace(':',' ').replace('/',' ').replace('_',' ').replace('--',' ').replace('[',' ').replace(']',' ').replace(' #',' ').replace('&',' ').replace('"',' ').replace('\uf0b7',' ').replace('–',' ').replace('“',' ').replace('”',' ').replace('•',' ').replace('\ufeff',' ').replace('’',' ').replace('|',' ') 
        resume_names.append(file)
        resume_texts.append(nlp(fus))
    return resume_texts, resume_names
    
def assess_education(education_dict):
    ed_high={}
    for key in education_dict:
        ed_high.setdefault(key, 0)
        if 'PHD' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.1
        if 'POSTGRADUATE' or 'MASTERS' or 'GRADUATE' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.08
        if 'BACHELORS' or 'UNDERGRADUATE' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.06
        if 'INTERMEDIATE' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.04
        if 'MATRICULATION' in education_dict[key]:
            ed_high[key]=ed_high[key]+0.02
        else:
            ed_high[key]=ed_high[key]+0
    return (ed_high)
       
def idf_tf_tech(cont):
    DIR = 'cvs txt/'
    sourcepath = os.listdir(DIR)
    num_cvs=len(sourcepath)
    wgt={}
    df={}
    idf={}
    tf={}
    dict={}
    index_tech_stem={}

    for key in index_tech:
        index_tech_stem.setdefault(key,[])
        for w in index_tech[key]:
            index_tech_stem[key].append(Pstem.stem(str(j)))
    # print(index_tech)
    #calculating df
    for w in V_tech:
        df.setdefault(w,0)
        for key in index_tech:
            if(w in index_tech[key]):
                df[w]+=1

    #calculating idf
    for key in df:
        idf.setdefault(key,0)
        # if(df[key]<=0):
        #     print(key)
        #     print(df[key])
        idf[key]=numpy.log2(num_cvs/df[key])
    
    #calculating tf
    for key in cont:
        dict.setdefault(key,{})
        for w in V_tech_stem:
            dict[key][w]=cont[key].count(w.lower())
        if(sourcepath[len(sourcepath)-1]==key):
            tf=dict
    idf_stem={}
    for key in idf:
        idf_stem.setdefault(Pstem.stem(str(key)),idf[key])

    #calculating tf*idf for each document
    for key in tf:
        wgt.setdefault(key,{})
        for w in V_tech_stem:
            wgt[key][w]=format(tf[key][w]*idf_stem[w],'.3F')
    return idf, wgt

def tech_JD_process(vacature_technicalskillset, idf, doc):
    wgt={}
    tf={}
    stemmer=[]
    for w in list(vacature_technicalskillset):
        stemmer.append(Pstem.stem(w))
    for w in V_tech_stem:
        tf.setdefault(w,0)
        if w in list(stemmer):
            tf[w]=doc.count(w)
    idf_stem={}
    for key in idf:
        idf_stem.setdefault(Pstem.stem(str(key)),idf[key])
    for w in V_tech_stem:
        wgt.setdefault(w,0)
        wgt[w]=format(tf[w]*idf_stem[w],'.3F')
    return(wgt)

def cos_sim_tech(d_wgt, q_wgt):
    cos={}
    for key in d_wgt:
        temp=[]
        temp1=[]
        for w in q_wgt:
            t=float(d_wgt[key][w])
            t1=float(q_wgt[w])
            temp.append(t)
            temp1.append(t1)           
        cos.setdefault(key,0)
        sum=0
        sumsqt=sumsqt1=0
        for w in range(len(temp)-1):
            sum=sum+temp[w]*temp1[w]
        sqt=[]
        sqt1=[]
        for w in range(len(temp)-1):
            sqt.append(temp[w]*temp[w])
            sqt1.append(temp1[w]*temp1[w])
        for w in range(len(temp)-1):
            sumsqt=sumsqt+sqt[w]
            sumsqt1=sumsqt1+sqt1[w]
        if math.sqrt(sumsqt)*math.sqrt(sumsqt1) ==0:
            cos[key]=0
        else:
            cos[key]=format(sum/(math.sqrt(sumsqt)*math.sqrt(sumsqt1)),'.3F')
    return cos

def idf_tf_soft(cont):
    DIR = 'cvs txt/'
    sourcepath = os.listdir(DIR)
    num_cvs=len(sourcepath)
    wgt={}
    df={}
    idf={}
    tf={}
    dict={}
    index_soft_stem={}

    for key in index_soft:
        index_soft_stem.setdefault(key,[])
        for w in index_soft[key]:
            index_soft_stem[key].append(Pstem.stem(str(j)))

    #calculating df
    for w in V_soft:
        df.setdefault(w,0)
        for key in index_soft:
            if(w in index_soft[key]):
                df[w]+=1
    
    #calculating idf
    for key in df:
        idf.setdefault(key,0)
        if(df[key]<=0):
            print(key)
            print(df[key])
        idf[key]=numpy.log2(num_cvs/df[key])

    #calculating tf
    for key in cont:
        dict.setdefault(key,{})
        for w in V_soft_stem:
            dict[key][w]=cont[key].count(w)
        
        if(sourcepath[len(sourcepath)-1]==key):
            tf=dict

    idf_stem={}
    for key in idf:
        idf_stem.setdefault(Pstem.stem(str(key)),idf[key])
    # print(tf)
    #calculating tf*idf for each document
    for key in tf:
        wgt.setdefault(key,{})
        for w in V_soft_stem:
            wgt[key][w]=format(tf[key][w]*idf_stem[w],'.3F')
    # print(wgt)
    return idf, wgt

def cos_sim_soft(d_wgt, q_wgt):
    cos={}
    for key in d_wgt:
        temp=[]
        temp1=[]
        for w in q_wgt:
            t=float(d_wgt[key][w])
            t1=float(q_wgt[w])
            # if(t1==0):
                # t1=0.000000001
            # if(t==0):
            #     t=0.000000001
            temp.append(t)
            temp1.append(t1)      
        # print('temp',temp)    
        # print('temp1',temp1)    
        cos.setdefault(key,0)
        sum=0
        sumsqt=sumsqt1=0
        for w in range(len(temp)-1):
            sum=sum+temp[w]*temp1[w]
        # print('mul',sum)
        sqt=[]
        sqt1=[]
        for w in range(len(temp)-1):
            sqt.append(temp[w]*temp[w])
            sqt1.append(temp1[w]*temp1[w])
        for w in range(len(temp)-1):
            sumsqt=sumsqt+sqt[w]
            sumsqt1=sumsqt1+sqt1[w]
        # print('sumsqt',sumsqt)
        # print('sumsqt1',sumsqt1)
        if math.sqrt(sumsqt)*math.sqrt(sumsqt1) ==0:
            cos[key]=0
        else:
            cos[key]=format(sum/(math.sqrt(sumsqt)*math.sqrt(sumsqt1)),'.3F')
        # cos[key]=format(1-spatial.distance.cosine(temp,temp1),'.3F')
    # print(cos)
    return cos

def soft_JD_process(vacature_softskillset, idf, doc):
    wgt={}
    tf={}
    stemmer=[]
    for w in list(vacature_softskillset):
        stemmer.append(Pstem.stem(w))
    for w in V_soft_stem:
        tf.setdefault(w,0)
        if w in list(stemmer):
            tf[w]=doc.count(w)
    idf_stem={}
    for key in idf:
        idf_stem.setdefault(Pstem.stem(str(key)),idf[key])
    for w in V_soft_stem:
        wgt.setdefault(w,0)
        wgt[w]=format(tf[w]*idf_stem[w],'.3F')
    return wgt

def combine_score(cos_tech, cos_soft, ed_high, vacature_education):
    total={}
    for key in cos_tech:
        total.setdefault(key,0)
        total[key]=float(cos_soft[key])+float(cos_tech[key])
    #######################
    for key in ed_high:
        total[key]=total[key]+ed_high[key]
    #######################
    return total

def show_result(total):
    final={}
    rejected={}
    alpha=0.3
    for key in total:
        if total[key] > alpha:
            final.setdefault(key,0)
            final[key]=total[key]
        else:
            rejected.setdefault(key,0)
            rejected[key]=total[key]
    final=sorted(final.items(), key =operator.itemgetter(1), reverse=True)
    print('ACCEPTED CVs')
    for key in final:
        print('\t'+key[0]+'\t'+str(format(key[1],'.3F')))
    print('REJECTED CVs')
    for key in rejected:
        print('\t'+key+'\t'+str(format(rejected[key],'.3F')))

    acc=[]
    c=0
    for key in final:
        acc.append([])
        acc[c].append(key[0])
        acc[c].append(key[1])
        c=c+1
    rej=[]
    c=0
    for key in rejected:
        rej.append([])
        rej[c].append(key)
        rej[c].append(rejected[key])
        c=c+1

    root = Tk()
    root.geometry('880x550')
    root.title("Skill Finder")
    label=['Accepted Resumes', 'Scores']
    count=len(acc)
    l=[]
    for j in range(count):
        l.append(j+1)

    txt = Text(root) 
    txt.pack() 

    class PrintToTXT(object): 
        def write(self, s): 
            txt.insert(END, s)

    sys.stdout = PrintToTXT()
    dframe = pd.DataFrame(acc, index=l, columns=label)
    print (dframe)
    print()
    print()
    count=len(rej)
    l=[]
    for j in range(count):
        l.append(j+1)
    label=['Rejected Resumes', 'Scores']
    dframe = pd.DataFrame(rej, index=l, columns=label)
    print (dframe)
    # exit

    mainloop()

def main():
    cont=create_index()                 #Preprocessing of resumes
    add_newruler_to_pipeline()          #Read training dataset
    Tk().withdraw()                     #we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename()        #Select job description file
    print(filename)

    doc=create_query_ind(filename)      #Preprocessing of JD
    print(doc)
    resume_texts, resume_names = create_tokenized_texts_list()          #Return Resumes content with their names
    f2=open(filename,'r')
    vacature_text=f2.read().lower()
    softskillset_dict, technicalskillset_dict, education_dict = create_skillset_dict(resume_names, resume_texts)

    global V_soft
    temp=[]
    for w in V_soft:
        if w not in temp:
            temp.append(w)
    V_soft=temp
    global V_soft_stem
    for w in V_soft:
        V_soft_stem.append(Pstem.stem(w))

    global V_tech
    temp=[]
    for w in V_tech:
        if w not in temp:
            temp.append(w)
    V_tech=temp
    global V_tech_stem
    for w in V_tech:
        V_tech_stem.append(Pstem.stem(w))

    idf_tech, wgt_tech=idf_tf_tech(cont)

    idf_soft, wgt_soft=idf_tf_soft(cont)

    vacature_softskillset = create_softskill_set_JD(nlp(vacature_text))

    vacature_technicalskillset = create_technicalskill_set_JD(nlp(vacature_text))

    vacature_education = create_education_set(nlp(vacature_text))

    tech_JD_wgt=tech_JD_process(vacature_technicalskillset, idf_tech, doc)

    cos_tech=cos_sim_tech(wgt_tech, tech_JD_wgt)

    soft_JD_wgt=soft_JD_process(vacature_softskillset, idf_soft, doc)

    cos_soft=cos_sim_soft(wgt_soft, soft_JD_wgt)

    ed_high=assess_education(education_dict)
    
    total=combine_score(cos_tech, cos_soft, ed_high, vacature_education)    

    show_result(total)

main()
