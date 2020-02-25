import pickle
import os
import re
from pycocotools.coco import COCO
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

class Vocab:
    '''
    Wrapper object for word vocabulary
    '''
    def __init__(self):
        '''
        Either load existing wordDict or create a new one if there is none
        '''
        
        dictFile = 'wordDict.pkl'
        statFile = 'wordStat.pkl'
        arrayFile = 'wordArray.pkl'
        
        wordDict = {}
        wordStat = {}
        wordArray = []
        
        nltk.download('punkt')

        # wordDict already exists
        if os.path.isfile(dictFile) and os.path.isfile(statFile) and os.path.isfile(arrayFile):
            with open(dictFile, 'rb') as f:
                wordDict = pickle.load(f)
            with open(statFile, 'rb') as f:
                wordStat = pickle.load(f)
            with open(arrayFile, 'rb') as f:
                wordArray = pickle.load(f)
            
            print("Successfully loaded vocabulary from {}, {} and {}".format(dictFile, statFile, arrayFile))
            
        # need to build wordDict
        else:
            coco = COCO('./data/annotations/captions_train2014.json')
            
            # Special words
            wordDict['<start>'] = 0
            wordDict['<end>'] = 1
            wordDict['<unknown>'] = 2
            wordArray = ['<start>', '<end>', '<unknown>']

            ids = list(coco.anns.keys())
            maxLength = 0
            
            # maxIndex for newly encountered words. Start at 3 since 0,1,2 are special words
            maxIndex = 3
            
            # Go over each annotation
            for i, idx in enumerate(ids):
                
                # Turn caption into list of words
                newCaption = str(coco.anns[idx]['caption'])
                
                tokens = word_tokenize(newCaption)
                words = [word for word in tokens if word.isalpha()]
                
                maxLength = max(maxLength, len(words))
                
                # Check each word, assign new index if newly encountered
                for word in words:
                    if (word not in wordDict):
                        wordDict[word] = maxIndex
                        wordArray.append(word)
                        maxIndex += 1

            
            # Stats to help determine one-hot-encoding and sequence size
            wordStat['vocab_size'] = maxIndex
            wordStat['max_length'] = maxLength
            
            with open(dictFile, 'wb') as f:
                pickle.dump(wordDict, f)
            with open(statFile, 'wb') as f:
                pickle.dump(wordStat, f)
            with open(arrayFile, 'wb') as f:
                pickle.dump(wordArray, f)
                
            print("Successfully generated vocabulary to {}, {} and {}".format(dictFile, statFile, arrayFile))
        
        
        self.wordDict = wordDict
        self.wordArray = wordArray
        self.vocabSize = wordStat['vocab_size']
        self.maxLength = wordStat['max_length']
        
        print("vocab size", self.vocabSize)
        print("max sequence", self.maxLength)
    
    def __call__(self, word):
        if (word not in self.wordDict):
            word = '<unknown>'

        return self.wordDict[word]
    
    def one_hot_encoding(self, labels):
        """
        Encode labels using one hot encoding and return them.
        """
        numClasses, maxLength = self.vocabSize, self.maxLength

        enc = np.zeros((maxLength, numClasses))#, maxLength))
        enc[np.arange(len(labels)), labels] = 1
        return enc

# def one_hot_decoding(enc, vocab):
    