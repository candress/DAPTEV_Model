# -*- coding: utf-8 -*-
# Code adapted from:
# Shubham Patel
# https://github.com/bayeslabs/genmol
# https://blog.bayeslabs.co/2019/06/04/All-you-need-to-know-about-Vae.html

import torch
import torch.nn as nn
import torch.nn.functional as F

encoder_bidir = True
vae_encoder_hidden = 256
vae_encoder_layers = 1
vae_encoder_dropout = 0.5
vae_decoder_layers = 3
vae_decoder_dropout = 0
d_z = 10
vae_decoder_hidden = 512
dataset = None


class VAE(nn.Module):
    def __init__(self, data, regress_choice = 0):
        super().__init__()
        global dataset 
        dataset = data
        self.vocabulary = dataset.vocabulary
        self.vector = dataset.vector
        # regression style choice: 0 = BCE (uses score threshold), 1 = RMSE (does not use score threshold), 2 = MSE (does not use score threshold)
        self.regress_choice = regress_choice

        # obtain the number (amount) of vocab and dimension of embeddings (e.g. 87 and 87)
        n_vocab, d_emb = len(self.vocabulary), self.vector.size(1)
        # creating an embedding layer, passing in size of vocab, size of embeddings, and the index with witch to pad.
        # returns an embedding layer with a n_vocab x d_emb matrix with pre-populated weights to train.
        self.x_emb = nn.Embedding(n_vocab, d_emb, dataset.char_index['<pad>'])
        # overwrite the pre-populated weights with the 1-hot vector
        self.x_emb.weight.data.copy_(self.vector)

        # Encoder
        self.encoder_model = nn.GRU(d_emb, vae_encoder_hidden, vae_encoder_layers, batch_first = True, dropout = vae_encoder_dropout if vae_encoder_layers > 1 else 0, bidirectional = encoder_bidir)
        encoder_last_layer_size = vae_encoder_hidden * (2 if encoder_bidir else 1)
        self.encoder_mu = nn.Linear(encoder_last_layer_size, d_z)
        self.encoder_logvar = nn.Linear(encoder_last_layer_size, d_z)

        # Decoder
        self.decoder_model = nn.GRU(d_emb + d_z, vae_decoder_hidden, num_layers = vae_decoder_layers, batch_first = True, dropout = vae_decoder_dropout if vae_decoder_layers > 1 else 0)
        self.decoder_latent = nn.Linear(d_z, vae_decoder_hidden)
        self.decoder_fullyc = nn.Linear(vae_decoder_hidden, n_vocab)

        # Score predictor and regularizer
        # Making layers in Neural Net (NN). fc = fully connected, 1 is the first layer
        # NOTE: there are 32 latent vectors (z) per forward pass and each z has d_z 'features', NN needs d_z inputs
        self.fc1 = nn.Linear(d_z, 64) 
        self.fc2 = nn.Linear(64, 32) 
        self.fc3 = nn.Linear(32, 1)

        # Grouping the model's parameters
        self.encoder = nn.ModuleList([self.encoder_model, self.encoder_mu, self.encoder_logvar])
        self.decoder = nn.ModuleList([self.decoder_model, self.decoder_latent, self.decoder_fullyc])
        self.vae = nn.ModuleList([self.x_emb, self.encoder, self.decoder])
        self.ann = nn.ModuleList([self.fc1, self.fc2, self.fc3])


    '''---------------------------------------------- HELPER FUNCTIONS -------------------------------------------------------------'''    
    
    @property
    def device(self):
        return next(self.parameters()).device
    

    def forward(self, x, scores):
        z, kl_loss = self.forward_encoder(x)
        scores = torch.tensor(scores, device = dataset.device).unsqueeze(1)
        # calculate the ANN output and compare it against the correct score
        prediction_loss = self.forward_score(z, scores)
        # pass data into decoder to get reconstruction loss
        recon_loss = self.forward_decoder(x, z)
        return kl_loss, prediction_loss, recon_loss
    

    def forward_encoder(self, x):
        mu, logvar, z = self.forward_encoder_mu_logvar_z(x)
        # calculate the dissimilarity between two distributions (KL divergence)
        kl_loss = 0.5 * (logvar.exp() + mu ** 2 - 1 - logvar).sum(1).mean()
        return z, kl_loss


    def forward_encoder_mu_logvar_z(self, x):
        # overwrite x (tuple of tensors of size batch) with a list of tensors that represent the embedding weights
        x = [self.x_emb(i_x) for i_x in x]
        # turn that list of tensor weights into a "packed sequence" for the RNN (reduces computation internally)
        x = nn.utils.rnn.pack_sequence(x)
        # pass the list of packed tensor weights into the GRU and obtain the final hidden state (not the putput). 
        # Note, hidden state is next prediction, output (concatenation of every hidden state time step) is often used in label classification prediction.
        _, h = self.encoder_model(x, None)
        # overwrite h with just the last 1D index to remove bidirection unless bidirection is allowed
        h = h[-(1 + int(self.encoder_model.bidirectional)):]
        # if bidirection is allowed, hidden layer will have 2 dimensions (forward and backward), split into 2 separate tensors, 
        # concatenate them about the last dimension, then squeeze to remove first 1 dimension. If not, nothing happens here
        h = torch.cat(h.split(1), dim=-1).squeeze(0)
        # get the mu and log variance by passing h through 2 different linear modules (with different weights)
        mu, logvar = self.encoder_mu(h), self.encoder_logvar(h)
        # eps (epsilon) will hold a random tensor from a normal distribution with 0 mean and 1 variance that is the same size as mu
        eps = torch.randn_like(mu)
        # divide vals in logvar by 2, raise e by resulting values, multiply by eps, add to mu (reparameterization trick)
        z = mu + (logvar / 2).exp() * eps
        return mu, logvar, z
    

    def forward_score(self, z, scores):
        # pass hidden layer into fully connected layers
        fc_output = self.fc1(z)
        fc_output = self.fc2(fc_output)
        regression = self.fc3(fc_output)
        # squeeze values between 0 and 1, BCE does not like negative values
        sig = torch.sigmoid(regression)
        # perform loss calculation (depending on regression type selected)
        # by default, theses calculate mean of batch size
        if self.regress_choice == 0:
            predict_loss = F.binary_cross_entropy(sig, scores)
        elif self.regress_choice == 1:
            predict_loss = torch.sqrt(F.mse_loss(sig, scores))
        else:
            predict_loss = F.mse_loss(sig, scores)

        return predict_loss

    
    def forward_decoder(self, x, z):
        x, y = self.forward_decoder_output(x, z)
        # by default, this calculates mean of batch size
        recon_loss = F.cross_entropy(y[:, :-1].contiguous().view(-1, y.size(-1)), x[:, 1:].contiguous().view(-1), ignore_index= dataset.char_index['<pad>'])
        return recon_loss


    def forward_decoder_output(self, x, z):
        lengths = [len(i_x) for i_x in x]
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value= dataset.char_index['<pad>'])
        x_emb = self.x_emb(x)
        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        x_input = torch.cat([x_emb, z_0], dim=-1)
        x_input = nn.utils.rnn.pack_padded_sequence(x_input, lengths, batch_first=True)
        h_0 = self.decoder_latent(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_model.num_layers, 1, 1)
        output, _ = self.decoder_model(x_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        y = self.decoder_fullyc(output)
        return x, y


    # return a latent vector with a random normal distribution (0 mean, 1 variance) of the same dimensions as the specified batch size and encoder mu output 
    def sample_z_prior(self, n_batch):
        return torch.randn(n_batch, self.encoder_mu.out_features, device= self.x_emb.weight.device)


    # Function to return samples from the decoder.
    def sample(self, z=None, n_batch=1, max_len=40, temperature=1.0):
        with torch.no_grad():
            if z is None:
                # z will have size of n_batch tensors x d_z long
                z = self.sample_z_prior(n_batch)
            
            z = z.to(self.device)
            # z_0 size = n_batch x 1 x d_z (z remains unchanged). So far, not based on encoder at all (except for what the decoder already learned from the encoder)!
            z_0 = z.unsqueeze(1)
            # Get hidden output from decoder latent layer (just a trained torch.Linear). vae_decoder_hidden is the funnel up dimension
            # input size = n_batch x d_z, output = n_batch x vae_decoder_hidden
            h = self.decoder_latent(z)
            # inject a dimension of 1 at the front, then repeat the tensor num_layers times (the 1s just mean copy 1 time), 
            # size num_layers x n_batch x vae_decoder_hidden
            h = h.unsqueeze(0).repeat(self.decoder_model.num_layers, 1, 1)
            # create a tensor with a 1D list containing one element (the index for <bos>), repeat n_batch times to fill list with that many elements
            w = torch.tensor(dataset.char_index['<bos>'], device=self.device).repeat(n_batch)
            # similar to w, but need <pad> index and repeat to make a 2D list of n_batch x max_len
            x = torch.tensor([dataset.char_index['<pad>']], device=dataset.device).repeat(n_batch, max_len)
            # slice the x list and return the 0 column (this only works for numpy and pytorch arrays). Set 0 column to <bos> index
            x[:, 0] = dataset.char_index['<bos>']
            # create a new 1D tensor filled with max_length number n_batch times
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
            # do similar, but filled with zeros
            eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)
            
            # Build a sequence per batch item up to length max_len (inclusive, exclusive)
            for i in range(1, max_len):
                # get embedding row for w (index tensor), with a 1 dim added in the 1 column of tensor (batch first). Start with BOS
                x_emb = self.x_emb(w).unsqueeze(1)
                # concatentate the x_embeddings with z_0 in the last dimension of the tensor, size = n_batch x 1 x (d_emb + d_z)
                x_input = torch.cat([x_emb, z_0], dim=-1)
                # pass x_input and h to the GRU decoder, get output (last hidden layer) and all previous hidden layers (final hidden state for the input sequence)
                # o size = n_batch x 1 x vae_decoder_hidden, h size = 3 x n_batch x vae_decoder_hidden
                o, h = self.decoder_model(x_input, h)
                # get output from torch.Linear, passing in GRU output, doing away with the 1 in the 2nd column. This will convert GRU output back to same size as vocab
                y = self.decoder_fullyc(o.squeeze(1))
                # perform softmax (divided by temperature) on y's last layer batch item (i.e. tell me which vocab index has been predicted per tensor in batch)
                y = F.softmax(y / temperature, dim=-1)
                # return tensor where each row contains the predicted vocab index based on the multinomial probability distribution being passed in (output size = n_batch x 1)
                # Basically, convert the probabilities returned by softmax into the predicted index per element in batch
                w = torch.multinomial(y, 1)
                # then slice this list and return the 0 column to make it a 1D array of size n_batch rather than a list of n_batch x 1 (n_batch 1D lists of size 1)
                w = w[:,0]
                # The tilde (~) symbol in python performs a bitwise negation operation (returns the complement of the bit value, bit flip, basically invert sign and subtract 1)
                # The eos_mask is of type uint8 which is an unsigned 8 bit integer. Meaning, performing the ~ operation on its 0 (false) values yields 255 (true) 
                # Because pytorch is like numpy, I can do 2D row, column referencing e.g. a[x, y]. Using the negation of eos_mask in the row index returns values where the negation
                # equates to true (255) and ignores the false (0) values
                # Summary, do a transpose on w and insert these elements along the column of the batch at element i where the w values equate to true
                x[~eos_mask, i] = w[~eos_mask]
                # convert the indices in eos_mask to 1 (the rest are 0s) where w == the index value for eos. In other words, note where 'eos' was predicted
                i_eos_mask = ~eos_mask & (w == dataset.char_index['<eos>'])
                # note the positions (i) of all the eos's over time in the end_pad tensor
                end_pads[i_eos_mask] = i + 1
                # update eos_mask to change the 0s to 255s (I think) if there is overlap between eos_mask and i_eos_mask
                eos_mask = eos_mask | i_eos_mask
                # now copy all elements from x into new_x
        new_x = []
        for i in range(x.size(0)):
            new_x.append(x[i, :end_pads[i]])
                        
        return [self.tensor2string(i_x) for i_x in new_x]


    
    # convert the tensor back to a string
    def tensor2string(self, tensor):
            ids = tensor.tolist()
            string = dataset.id_to_string(ids, rem_bos=True, rem_eos=True)
            return string