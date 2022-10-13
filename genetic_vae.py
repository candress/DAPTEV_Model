from model.data import Data
from model.samples import sample
from model.trainer import Trainer
from model.vae_model import VAE
from model.ga_operations import GA_Ops
from tqdm import tqdm

class DAPTEV:
    def __init__(self, epochs=30, batch=32, combination_amnt=3, score_threshold=5000, n_samples=100, max_seq_len=40, tourn_select_k=2, elite_percent=0.01, 
                crossover_rate=0.9, mutation_rate=0.01, selection_rate=0.95, starting_learn_rate = 0.003, ending_learn_rate = 3 * 1e-4, 
                starting_KL_weight = 0.001, ending_KL_weight = 0.1, regress_choice = 0):

        self.n_epochs = epochs
        self.batch = batch
        self.combination_amount = combination_amnt
        self.score_threshold = score_threshold
        self.n_returned_samples = n_samples
        self.max_seq_len = max_seq_len
        self.tourn_select_k = tourn_select_k
        self.elite_percent = elite_percent
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.select_rate = selection_rate
        self.start_lr = starting_learn_rate
        self.end_lr = ending_learn_rate
        self.start_KL_weight = starting_KL_weight
        self.ending_KL_weight = ending_KL_weight
        self.regress = regress_choice
        self.model = None

        tqdm.write("building data")
        self.d = Data(combinations = combination_amnt)
        tqdm.write("creating model and providing vocab & vector from Data class")
        self.model = VAE(self.d).to(self.d.device)
        self.my_trainer = Trainer(self.d, self.n_epochs, self.batch, self.score_threshold, self.start_lr, self.end_lr, self.start_KL_weight, self.ending_KL_weight, self.regress)
        

    def update_data(self, data_frame):
        self.d.update_training_data(data_frame)


    def train_model(self, data_frame, ga_only = False, vae_writer = None):
        tqdm.write("updating training data")
        self.update_data(data_frame)
        if not ga_only:
            tqdm.write("setting model to train mode")
            self.model.train()
            tqdm.write("fitting the model with the training data\n")
            self.my_trainer.fit(self.model, self.d.training_data, summary_writer = vae_writer)
        # Calling "flush" to make sure all pending events have been written to disk
            vae_writer.flush()
        # don't need vae writer anymore
            vae_writer.close()
            tqdm.write("setting model to evaluation mode")
            self.model.eval()
        
        tqdm.write("performing selection")
    # copy data so I don't sort original data
        copy_data = []
        copy_data.extend(self.d.training_data)
    # Sort data based on score (smallest to largest)
        sorted_data = sorted(copy_data, key=lambda x: x[1])
    # perform tournament selection
        n_samples = min(len(sorted_data), self.n_returned_samples)
        elite = int(n_samples * self.elite_percent)
        # must have at least 1 elite
        elite = max(elite, 1)
        # obtain tournament-selected parents (new list of tuples - seq,score)
        parents = GA_Ops.tournament_selection(sorted_data, n_samples, elite, self.tourn_select_k, self.select_rate)
    # obtain all the data from dataframe for sequences to return back to the caller
        sorted_complete_data = self.d.get_sequence_data_list([seq for seq, _ in sorted_data])
    # generate new sequences based on selected sequence that have been encoded, had genetic operations performed on their latent representations, then decoded
        tqdm.write("grabbing training samples\n")
        # specifically choosing to use "parents" and not sorted because the selection algorithm shuffles for me and that order is important for crossover. 
        # Otherwise, crossover and mutation become less diverse and thus become a hillclimber algorithm
        children = sample.take_data_samples(self.model, self.my_trainer, self.d, parents, self.batch, self.n_returned_samples, self.max_seq_len, self.crossover_rate, self.mutation_rate, ga_only)
    # returning samples and all data associated with SORTED data (this way the caller can simply append the new data to an already sorted list)
        return children, sorted_complete_data