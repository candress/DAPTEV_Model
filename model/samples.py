# -*- coding: utf-8 -*-
# Code adapted from:
# Shubham Patel
# https://github.com/bayeslabs/genmol
# https://blog.bayeslabs.co/2019/06/04/All-you-need-to-know-about-Vae.html
from tqdm import tqdm
from model.ga_operations import GA_Ops

class sample():
    # method to produce new samples using training data and genetic operators
    def take_data_samples(model, trainer, data_class, data, n_batch, n_samples, max_len, crossover = 0.9, mutation = 0.01, ga_only = False):

        # convert data like I did in trainer.py for my encoder
        train_loader = trainer.get_dataloader(model, data)
        
        samples = []
        with tqdm(total=n_samples, desc='Generating samples') as T:
            for input_batch, _ in train_loader:
                # Iterate through each tensor in batch and send them to their device.
                # Then convert the entire input batch back into to a tuple of tensors.
                input_batch = tuple(seq.to(model.device) for seq in input_batch)
                n_batch = len(input_batch)
                if ga_only:
                    children = GA_Ops.cross_n_mutate(input_batch, crossover, mutation, ga_only, data_class)
                    # tensors to list
                    children = [ids.tolist() for ids in children]
                    # ids to strings
                    children = [data_class.id_to_string(ids, rem_bos=True, rem_eos=True) for ids in children]
                    # TODO the data in current_samples should be converted back to sequences
                    current_samples = children
                else:
                    # get z from encoder
                    _, _, z = model.forward_encoder_mu_logvar_z(input_batch)
                    # perform crossover and mutation
                    children_z = GA_Ops.cross_n_mutate(z, crossover, mutation, ga_only)
                    # get samples (no z means generate z from normal distribution, mean 0 variance 1)
                    current_samples = model.sample(children_z, n_batch, max_len)
                samples.extend(current_samples)
                # update tqdm bar based on current length of samples
                T.update(len(current_samples))
        return samples



    # method to produce samples from a normal distribution (0 mean, 1 variance) based solely on the trained decoder
    def take_generated_samples(model, n_samples, n_batch, max_len):
        n = n_samples
        samples = []
        with tqdm(total=n_samples, desc='Generating samples') as T:
            while n > 0:
                current_samples = model.sample(min(n, n_batch), max_len)
                samples.extend(current_samples)
                n -= len(current_samples)
                T.update(len(current_samples))
        return samples