# Code adapted from:
# Shubham Patel
# https://github.com/bayeslabs/genmol
# https://blog.bayeslabs.co/2019/06/04/All-you-need-to-know-about-Vae.html
import pandas as pd
import torch
from itertools import product

class Data:
    def __init__(self, combinations = 1):
        # if combinations is outside of 1-3 break. This is not implimented
        if combinations > 3 or combinations < 1:
            exit("\nCombinations must be a number between 1 and 3 (inclusive). Exiting program.")
        
        # self.update_training_data(data_frame)

        # set() only enters unique data and turns it into a set
        characters = set()
        # the "update" method takes an iterable (string in this case) and adds items from the iterable to the set
        # because iterating over a string involves looking at each character, only unique characters will enter the set
        characters.update('cgua')

        # make a sorted list from the characters
        begin_chars = sorted(list(characters))
        # make a list of extras: "Beginning of Sequences", "End of Sequence", "Padding" (to make all sequences same length)
        extras = ['<bos>', '<eos>', '<pad>']
        self.max_char_len = 1

        # Add in combinations of bases to increase vocabulary and search space. Note, This can be done with a for loop.
        # I chose to do it this way due to the simplicity of implimentation. But this is way is not scalable.
        all_chars = None
        if combinations == 2:
            all_prods2 = [''.join(p) for p in product(begin_chars, begin_chars)]
            all_chars = begin_chars + all_prods2 + extras
            self.max_char_len = 2
        if combinations == 3:
            all_prods2 = [''.join(p) for p in product(begin_chars, begin_chars)]
            all_prods3 = [''.join(p) for p in product(all_prods2, begin_chars)]
            all_chars = begin_chars + all_prods2 + all_prods3 + extras
            self.max_char_len = 3
        
        self.vocabulary = all_chars
        # model character to ids, and ids to character. In other words:
        # grab each character and associate with a number, and the same vice versa in a dictionary
        self.char_index = {c: i for i, c in enumerate(all_chars)}
        self.index_char = {i: c for i, c in enumerate(all_chars)}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # create an identity matrix (for 1-hot embeddings)
        self.vector = torch.eye(len(self.char_index))



        '''---------------------------------------------- HELPER FUNCTIONS -------------------------------------------------------------'''

    def update_training_data(self, data_frame):        
        # copy only the portion of the dataframe where the "Split" column == "train"
        train_data = data_frame[data_frame['Split'] == 'train']
        
        # copy only the Sequences and Scores column of the previous dataframe and convert to list.
        train_data_seq = (train_data["Sequences"].squeeze()).astype(str).tolist()
        train_data_scores = (train_data["Scores"].squeeze()).astype(float).tolist()
        # associate the scores with their sequences again in a 2D array
        self.training_data = list( zip(train_data_seq, train_data_scores) )
        # Make a dictionary to allow for a speedy lookup of sequences to the rest of the associated dataframe values (must 'T'ranspose the dataframe to do this)
        df_copy_seq_index = data_frame.copy().set_index('Sequences')
        self.all_data_dict = df_copy_seq_index.T.to_dict('list')
        

    def get_sequence_data_list(self, data):
        '''
            Returns a 2D list with all of the information associated with the provided sequences
            
            Paramters:
                data: a list of sequences (str)

            Returns:
                a 2D list where each sub list has the format [Sequence (str), Secondary Structure (str), Score (float), Test/Train Split (str)]
        '''
        return [self.get_sequence_data(seq) for seq in data]

    
    def get_sequence_data(self, sequence):
        '''
            Returns the associated data from the dataframe for a particular sequence (including the sequence itself) as a list.

            Parameters: 
                sequence: the sequence for which to obtain the associated data
            Returns:
                a list of format [Sequence (str), Secondary Structure (str), Score (float), Test/Train Split (str)]
        '''
        return [sequence] + self.all_data_dict[sequence]
    
        
    # return the index number for the specified character in the dictionary
    def char_to_index(self,char):
        return self.char_index[char]


    # same as "char_to_index", but reverse
    def index_to_char(self,id):
        return self.index_char[id]


    # convert strings (sequences) to their associated indices in the dictionary.
    # Return the list of indices
    def string_to_id(self, text, add_bos=False, add_eos=False):
        # make a new list and fill it with index numbers for each character set (of size 1/2/3 depending on choice) in the dictionary
        characters = list(text)
        substrings = []
        while len(characters) > 0 :
            if len(characters) > self.max_char_len:
                substr = ''.join(characters[:self.max_char_len])
                del characters[:self.max_char_len]
                substrings.append(substr)
            else:
                substrings.append(''.join(characters))
                del characters[:len(characters)]

        # bps stands for base pairs
        ids = [self.char_to_index(bps) for bps in substrings]

        if add_bos:
            ids = [self.char_index['<bos>']] + ids
        if add_eos:
            ids = ids + [self.char_index['<eos>']]
        return ids


    # Same as "string_to_id", but reverse
    def id_to_string(self, ids, rem_bos=True, rem_eos=True):
        if len(ids) == 0:
            return ''
        bos = self.char_index['<bos>']
        eos = self.char_index['<eos>']
        if rem_bos:
            # ids = ids[1:]
            ids = list(filter(lambda id: id != bos, ids))
        if rem_eos:
            # ids = ids[:-1]
            ids = list(filter(lambda id: id != eos, ids))
        string = ''.join([self.index_to_char(id) if id != self.char_index['<pad>'] else '' for id in ids])
        return string


    # Send the target (score) to tensor on specified device and return it.
    def target_to_tensor(self, number, device='model'):
        tensor = torch.tensor(number, dtype=torch.float, device=self.device if device == 'model' else device)
        return tensor
    
    
    # Build a tensor of ids for the associated string, cast that to a tensor on specified device, and return it.
    def string_to_tensor(self, string, device='model'):
        ids = self.string_to_id(string, add_bos=True, add_eos=True)
        tensor = torch.tensor(ids, dtype=torch.long, device=self.device if device == 'model' else device)
        return tensor


    

        


    