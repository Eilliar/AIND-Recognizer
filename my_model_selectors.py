import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

DEBUG = False
class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    # Note about implementation:
    # I got a bit confused on the calculation of p. So I'm following the instructions on Udacity's forum:
    # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/12
    # Where alvaro_257625 highligthed:
    # Initial state occupation probabilities (isop) = numStates
    # Transition probabilities (tp) = numStates*(numStates - 1)
    # Emission probabilities (ep) = numStates*numFeatures*2 = numMeans+numCovars
    # p = Initial state occupation probabilities + Transition probabilities + Emission probabilities
    # for a model trained as follows (base_model): 
    # GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000, random_state=self.random_state)
    
    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        # Variable initialions
        best_bic_score = float("+inf") # since we have to minimize it, make sense to start from +inf
        best_model = None
        n_features = self.X.shape[1]

        # iterate over the number of hidden states (num_hidden_states)
        for n_hidden_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Build model
                model = self.base_model(n_hidden_states)
                # Compute loglikelihood
                logLikelihood = model.score(self.X, self.lengths)
                # Compute log(N), where N = number of data points.
                logN = np.log(len(self.X))
                # Compute p
                isop = n_hidden_states
                tp = n_hidden_states*(n_hidden_states - 1)
                ep = n_hidden_states*n_features*2
                p =  isop + tp + ep
                # BIC
                bic_score = -2*logLikelihood + p*logN
                if DEBUG:
                    print("BIC: {}".format(bic_score))

                best_bic_score, best_model =  min((best_bic_score, best_model), (bic_score, model))
            except Exception as e:
                if DEBUG:
                    # Print some debug information
                    print('ERROR: {}'.format(e))
                pass

        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_dic_score = float("-inf")
        best_model = None
        n_features = self.X.shape[1]

        # iterate over the number of hidden states (num_hidden_states)
        for n_hidden_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n_hidden_states)
                logLikelihood = model.score(self.X, self.lengths)

                #Calculate all other words average score
                p_score = 0
                count = 0
                for word in self.words:
                    if word != self.this_word:
                        new_X, new_lengths = self.hwords[word]
                        try:
                            p_score += hmm_model.score(new_X, new_lengths)
                            count += 1
                        except:
                            pass
                if count > 0:
                    logAllButword = p_score/count
                else:
                    logAllButword = 0

                #Calculate the total score
                dic_score = logLikelihood - logAllButword
                
                best_score, best_model = max((best_dic_score,best_model),(dic_score,model))
            
            except Exception as e:
                if DEBUG:
                    # Print some debug information
                    print('ERROR: {}'.format(e))
                pass

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        # Variable initializations
        best_model = None
        best_num_components = self.min_n_components
        best_logLikelihood = float('-inf') # since we have to maximize it, make sense to start from -inf

        # Handle case when have only one sequence:
        if len(self.sequences) == 1:
            for n_hidden_states in range(self.min_n_components, self.max_n_components + 1):
                p_score = 0
                try:
                    model = None
                    X_train, train_lengths = combine_sequences([0], self.sequences)
                    # It doesn't make sense to try CV with one sequence, but if so we can use the same sequence from train
                    X_test, test_lengths = combine_sequences([0], self.sequences)
                    # Train GassianHMM
                    model = GaussianHMM(n_components = n_hidden_states, covariance_type="diag", n_iter = 1000, 
                        random_state=self.random_state, verbose = False).fit(X_train, train_lengths)
                    # Score the trained model on the test set
                    p_score = model.score(X_test, test_lengths)

                except Exception as e:
                    if DEBUG:
                        # Print some debug information
                        print('ERROR: Training HMM. \ncv_train_idx: {}\tcv_test_ix:{}\nError message: {}'.format(cv_train_idx, 
                            cv_test_idx, e))
                    pass

                if p_score > best_logLikelihood:
                    best_logLikelihood, best_num_components, best_model = p_score, n_hidden_states, model

            return best_model

        # Cases when number os sequences >= 2
        # Define method and number of splits to perform
        n_splits = min(len(self.sequences), 5)
        if DEBUG:
            print("len(sequence): {}\tn_splits: {}".format(len(self.sequences), n_splits))
        split_method = KFold(n_splits=n_splits)
        # iterate over the number of hidden states (num_hidden_states)
        for n_hidden_states in range(self.min_n_components, self.max_n_components + 1):
            p_score = 0
            # number of trained models, to use on score average
            i = 0

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                # Default Model is None, to return in case of error.
                model = None
                # Try to train model, it's know hmmlearn to raise errors sometimes.
                try: 
                    if len(self.sequences) > 2:
                        # Combine training set
                        X_train, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                        # Combine test set
                        X_test, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                    else: # try to handle error cases like the ones that occur on word FISH
                        X_train, train_lengths = combine_sequences([0], self.sequences)
                        if len(self.sequences) == 2:
                            X_test, test_lengths = combine_sequences([1], self.sequences)
                        else:
                            # It doesn't make sense to try CV with one sequence, but if so we can use the same sequence from train
                            X_test, test_lengths = combine_sequences([0], self.sequences)

                    # Train GaussianHMM
                    model = GaussianHMM(n_components = n_hidden_states, covariance_type="diag", n_iter = 1000, 
                        random_state=self.random_state, verbose = False).fit(X_train, train_lengths)
                    # Score the trained model on the test set
                    p_score = model.score(X_test, test_lengths)
                    i += 1
                except Exception as e:
                    if DEBUG:
                        # Print some debug information
                        print('ERROR: Training HMM. \ncv_train_idx: {}\tcv_test_ix:{}\nError message: {}'.format(cv_train_idx, 
                            cv_test_idx, e))
                    pass

            if i > 0:
                # Average score
                n_score = p_score / i
            else:
                n_score = 0

            if n_score > best_logLikelihood:
                best_logLikelihood, best_num_components, best_model = n_score, n_hidden_states, model

        return best_model