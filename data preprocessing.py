def preprocess(file):
    """
    Description:
     this function get text files as input and outputs triples (context, question, answer)

     inputs:
     file : a .txt file that contains train text

     :returns:
        data: a list of [context, question, answer, support]s , context: a list of sentences
     """
    
    with open(file, 'r') as file:
        data=[]
        context=[]
        for line in file:
            line=line.lower()
            number, text = tuple(line.strip().split(" ", 1))
            if number is "1":
                ''' new context '''
                context = []
            if "\t" in text:
                '''Tabs are the separator between questions and answers, and are not present in context statements'''
                question, answer, support = tuple(text.split("\t"))
                if len(answer.split(","))==1: 
                    ''' we only accept those that have 1-word answer'''
                    data.append([context.copy(),question, answer,[int(s) for s in support.split()]])
            else:
                context.append(text)
        return data

PAD_TOKEN = '_PAD'
PAD_ID = 0
from nltk.tokenize import word_tokenize

def tokenize(sentence):
    "Tokenize a string by splitting on non-word characters and stripping whitespace."
    words = word_tokenize(sentence)
    return [word.lower() for word in words]

def parse_stories(file, only_supporting=False):
    """
    Parse the bAbI task format described here: https://research.facebook.com/research/babi/
    If only_supporting is True, only the sentences that support the answer are kept.
    """
    stories = []
    story = []
    lines=open(file,'r')
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            query, answer, supporting = line.split('\t')
            query = tokenize(query)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            stories.append((substory, query, answer))
            story.append('')
        else:
            sentence = tokenize(line)
            story.append(sentence)
    return stories
  
def get_tokenizer(data):
    "Recover unique tokens as a vocab and map the tokens to ids."
    tokens_all = []
    for story, query, answer in data:
        tokens_all.extend([token for sentence in story for token in sentence] + query + [answer])
    vocab = [PAD_TOKEN] + sorted(set(tokens_all))
    token_to_id = {token: i for i, token in enumerate(vocab)}
    return vocab, token_to_id
  
  
def tokenize_stories(parsed_data,token_to_id):
    story_ids=[]
    max_sent_num=0
    max_sent_len=0
    for context, query, answer in parsed_data:
        if len(context)>max_sent_num:
          max_sent_num=len(context)
        for sentence in context:
          if len(sentence)>max_sent_len:
            max_sent_len=len(sentence)
        context = [[token_to_id[token] for token in sentence] for sentence in context]
        query = [token_to_id[token] for token in query]
        if len(query)>max_sent_len:
          max_sent_len=len(query)
        answer = token_to_id[answer]
        story_ids.append((context[max(0,len(context)-130):], query, answer))
    return story_ids, max_sent_num, max_sent_len
  
def convert_to_tensors(tokenized_data, max_sent_num, max_sent_len):
  prgrphs_num=len(tokenized_data)
  print('prgrphs_num', len(tokenized_data))
  paragraphs=np.zeros(shape=[prgrphs_num, max_sent_num, max_sent_len],dtype=np.int32)
  paragraphs_mask=np.zeros(shape=[prgrphs_num, max_sent_num, max_sent_len],dtype=np.bool)
  questions=np.zeros(shape=[prgrphs_num, max_sent_len], dtype=np.int32)
  answers=np.zeros(shape=[prgrphs_num, 1],dtype=np.int32)
  i=0
  for prgrph, question, answer in tokenized_data:
    questions[i,:len(question)]=np.asarray(question)
    answers[i]=np.asarray(answer)
    for j in range(len(prgrph)):
      sentence=prgrph[j]
      paragraphs[i,j,:len(sentence)]=np.asarray(sentence)
      paragraphs_mask[i,j,:len(sentence)]=np.ones(shape=[len(sentence)],dtype=np.bool)
    i=i+1
    
  return paragraphs, paragraphs_mask, questions, answers
  

import spacy

def extract_keys(paragraphs,dictionary, n=20):

  keys_tags = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl","dobj", "dative", "attr", "oprd","pobj"]
  parser=spacy.load('en')
  keys=np.zeros([len(paragraphs),n],np.int32)
  keys_mask=np.zeros([len(paragraphs),n],np.bool)
  j=0
  for prgrph in paragraphs:
    cnt=0
    words_ind=set()
    doc=parser(prgrph)
    for i,tok in enumerate(doc):
        if tok.dep_ in keys_tags:
            words_ind.add(dictionary[str(tok)])
            cnt+=1
        if cnt==20:
            break
    keys[j,:len(words_ind)]= list(words_ind)
    keys_mask[j,:len(words_ind)]=np.ones([len(words_ind)],np.bool)
    j=j+1
    if j%1000==0:
      print('j:',j)
    
  return keys,keys_mask
