import torch
import random

class GA_Ops:
    def tournament_selection(data, n_samples, elite, k, select):
        '''
            This method performs tournament selection and assumes the data is sorted by best in decending order (best in position 0). 
            This method performs data copyint and shuffling
        '''
        data_copy = []
        data_copy.extend(data)
        parents = []
        data_size = len(data_copy)
        # perform elitism selection, add to parent list
        for i in range(elite):
            # note, data is already sorted with best score at index 0
            parents.append(data_copy[0])

        random.shuffle(data_copy)
        # perform tournament selection on remaining data (n_samples - elite times), add to parent list, remove from current list
        remaining = n_samples - elite
        for x in range(remaining):
            temp = []
            # for size of remaining tournament selection amount
            for y in range(k):
                # get a random index
                index = random.randrange(data_size)
                # grab the value at index in data and append it to temp list
                temp.append(data_copy[index])

            # get the index of the smallest score within temp (based on probability)
            sorted(temp, key=lambda x: x[1])
            for i,_ in enumerate(temp):
                rand_num = random.random()
                p = select*((1-select)**i)
                if rand_num < p:
                    # add the value in temp at index to parents
                    parents.append(temp[i])
                    break
            else:
                # remember that for else means the for loop did not encounter a break statement
                parents.append(temp[-1])
                
            # # get the index of the smallest score within temp
            # index = temp.index(min(temp, key = lambda x: x[1]))
            # # add the value in temp at index to parents
            # parents.append(temp[index])

        return parents


    def cross_n_mutate(parent_data, cross_rate, mutation_rate, ga_only = False, data_class = None):
        # TODO what if the sequences are different lengths. Need to pad and remove pad before proceeding
        # copy parent_data, turning the 1 tensor of a 2D list into a list of 1D tensors
        data_copy = [i for i in parent_data]
        if ga_only:
            data_copy = torch.nn.utils.rnn.pad_sequence(data_copy, batch_first=True, padding_value= data_class.char_index['<pad>'])
            # turn back into a list of tensors that now have their sequences padded
            data_copy = [i for i in data_copy]
        children = []
        while len(data_copy) > 0:
            p_1 = data_copy.pop(0)
            p_2 = data_copy.pop(0)

            # perform Uniform Order Cross Over
            c_1, c_2 = GA_Ops.uox(p_1, p_2, cross_rate)
            
            # perform mutation
            GA_Ops.mutate(c_1, mutation_rate, ga_only, data_class)
            GA_Ops.mutate(c_2, mutation_rate, ga_only, data_class)
            
            children.append(c_1)
            children.append(c_2)

        children = torch.stack(children)
        return children


    def uox(parent_1, parent_2, crossover_rate):
        # parents are 1D tensors
        child_1 = parent_1.clone()
        child_2 = parent_2.clone()
        rand_num = random.random()
        if rand_num < crossover_rate:
            # create a tensor of same size as parent_1 with random integers (< 2, >= 0) stored as boolean values
            mask = torch.randint_like(parent_1, high=2, low=0, dtype=bool)
            child_1[mask] = parent_2[mask]
            child_2[mask] = parent_1[mask]

        return child_1, child_2


    def mutate(child, mutation_rate, ga_only, data_class):
        rand_num = random.random()
        # rand_num = 0.001
        if rand_num < mutation_rate:
            if ga_only:
                copy_of_vocab = [chars for chars in data_class.vocabulary if chars != '<bos>' and chars != '<eos>' and chars != '<pad>']
                length = len(copy_of_vocab)
                index_of_vocab_char = random.randrange(length)
                length = len(child)
                index = random.randrange(length)
                # overwrite a random element within the 1D tensor with the random float
                child[index] = index_of_vocab_char
            else:
                # create a tensor filled with random floats from a normal distribution (but only size 1 x 1 = 1 2D list with 1 number) 
                rand_norm_dist_float = torch.randn(1, 1, device=child.device)
                length = len(child)
                index = random.randrange(length)
                # overwrite a random element within the 1D tensor with the random float
                child[index] = rand_norm_dist_float[0]


    


   