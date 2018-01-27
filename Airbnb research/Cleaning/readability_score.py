
# coding: utf-8

# In[144]:


#!/usr/bin/env python

import math
import pandas as pd
from utils import get_char_count
from utils import get_words
from utils import get_sentences
from utils import count_syllables
from utils import count_complex_words

class Readability:
    analyzedVars = {}

    def __init__(self, text):
        self.analyze_text(text)

    def analyze_text(self, text):
        words = get_words(text)
        char_count = get_char_count(words)
        word_count = len(words)
        sentence_count = len(get_sentences(text))
        syllable_count = count_syllables(words)
        complexwords_count = count_complex_words(text)
        avg_words_p_sentence = word_count/sentence_count
        
        self.analyzedVars = {
            'words': words,
            'char_cnt': float(char_count),
            'word_cnt': float(word_count),
            'sentence_cnt': float(sentence_count),
            'syllable_cnt': float(syllable_count),
            'complex_word_cnt': float(complexwords_count),
            'avg_words_p_sentence': float(avg_words_p_sentence)
        }

    def ARI(self):
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 4.71 * (self.analyzedVars['char_cnt'] / self.analyzedVars['word_cnt']) + 0.5 * (self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt']) - 21.43
        return score
        
    def FleschReadingEase(self):
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 206.835 - (1.015 * (self.analyzedVars['avg_words_p_sentence'])) - (84.6 * (self.analyzedVars['syllable_cnt']/ self.analyzedVars['word_cnt']))
        return round(score, 4)
        
    def FleschKincaidGradeLevel(self):
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.39 * (self.analyzedVars['avg_words_p_sentence']) + 11.8 * (self.analyzedVars['syllable_cnt']/ self.analyzedVars['word_cnt']) - 15.59
        return round(score, 4)
        
    def GunningFogIndex(self):
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.4 * ((self.analyzedVars['avg_words_p_sentence']) + (100 * (self.analyzedVars['complex_word_cnt']/self.analyzedVars['word_cnt'])))
        return round(score, 4)

    def SMOGIndex(self):
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (math.sqrt(self.analyzedVars['complex_word_cnt']*(30/self.analyzedVars['sentence_cnt'])) + 3)
        return score

    def ColemanLiauIndex(self):
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (5.89*(self.analyzedVars['char_cnt']/self.analyzedVars['word_cnt']))-(30*(self.analyzedVars['sentence_cnt']/self.analyzedVars['word_cnt']))-15.8
        return round(score, 4)

    def LIX(self):
        longwords = 0.0
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt'] + float(100 * longwords) / self.analyzedVars['word_cnt']
        return score

    def RIX(self):
        longwords = 0.0
        score = 0.0 
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = longwords / self.analyzedVars['sentence_cnt']
        return score
        


# In[145]:


data_listing=pd.read_csv('/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/airbnb_sen_merged.csv')
df_imported = pd.DataFrame(data_listing)


df_new=df_imported[["id","name","summary","space","description","neighborhood_overview","notes","transit","access","interaction","house_rules","host_about"]]
df_new


# In[146]:


df_readability=df_new[["id","name"]]
df_readability


# In[147]:


if __name__ == "__main__":
    for i in range(1,len(df_new.columns)):
        a=[];b=[];c=[];d=[];e=[];f=[];g=[];h=[];l=[];
        df_a=df_new.iloc[:,[0,i]].copy()
        for j, row in df_a.iterrows():
            text=row[str(df_a.columns[1])]
            try: 
                string=str(text).decode('utf-8')
                rd =Readability(string)
                h.append(rd.ARI())
                a.append(rd.FleschReadingEase())
                b.append(rd.FleschKincaidGradeLevel())
                c.append(rd.GunningFogIndex())
                d.append(rd.SMOGIndex())
                e.append(rd.ColemanLiauIndex())
                f.append(rd.LIX())
                g.append(rd.RIX())
                l.append(row['id'])
            except:
                h.append("")
                a.append("")
                b.append("")
                c.append("")
                d.append("")
                e.append("")
                f.append("")
                g.append("")
                l.append("")
                    #dataframe
        
        #df_readability.reset_index()
        df_output = pd.DataFrame({str(df_a.columns[1])+'_ARI': h,str(df_a.columns[1])+'_FleschReadingEase' :a,str(df_a.columns[1])+'_FleschKincaidGradeLevel':b,str(df_a.columns[1])+'_GunningFogIndex':c,str(df_a.columns[1])+'_SMOGIndex':d,str(df_a.columns[1])+'_ColemanLiauIndex':e,str(df_a.columns[1])+'_LIX':f,str(df_a.columns[1])+'_RIX':g})
        #df_readability=pd.merge(df_output,df_readability, on='id', how='inner') 
        #df_readability = df_readability.append(df_output)
        #df_readability.reset_index()
        df_readability = pd.concat([df_readability , df_output], axis=1)
        print(df_readability.shape)
    df_readability


# In[148]:


df_readability


# In[149]:


df_readability.to_csv("/Users/jaideep/Desktop/Airbnb Project Files/Data/new_data/readability.csv", sep=',')

