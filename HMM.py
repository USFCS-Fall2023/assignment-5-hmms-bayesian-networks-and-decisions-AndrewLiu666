import random
import argparse
import codecs
import os, sys
import numpy as np
import random


# AndrewLiu666 - Andrew Liu

# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        transitions = basename + ".trans"
        emissions = basename + ".emit"
        with open(transitions, 'r') as f:
            for line in f:
                list = line.strip().split()
                if list[0] not in self.transitions:
                    self.transitions[list[0]] = {}
                self.transitions[list[0]][list[1]] = float(list[2])

        with open(emissions, 'r') as f:
            for line in f:
                list = line.strip().split()
                if list[0] not in self.emissions:
                    self.emissions[list[0]] = {}
                self.emissions[list[0]][list[1]] = float(list[2])
        print(self.transitions)
        print(self.emissions)

    ## you do this.
    # https: // www.w3schools.com / python / ref_random_choices.asp
    # https: // stackoverflow.com / questions / 4859292 / how - can - i - get - a - random - key - value - pair -from-a - dictionary
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        state = '#'
        observation_type = ''
        observation_content = ''
        for i in range(n):
            next_type = random.choices(
                population=list(self.transitions[state].keys()),
                weights=list(self.transitions[state].values()),
                k=1
            )[0]
            observation_type += next_type
            observation_type += ' '
            emission = random.choices(
                population=list(self.emissions[next_type].keys()),
                weights=list(self.emissions[next_type].values()),
                k=1
            )[0]
            observation_content += emission
            observation_content += ' '
        print(observation_type)
        print(observation_content)

    ## you do this: Implement the Viterbi alborithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def forward(self, observations):
        n = len(observations)
        states = list(self.transitions.keys())
        M = np.zeros((len(states), n))
        for s in states:
            if s != '#':
                M[states.index(s), 0] = self.emissions[s].get(observations[0], 0) * self.transitions['#'].get(s, 0)

        for i in range(1, n):
            for s in states:
                sum_prob = 0
                for s2 in states:
                    if s != '#':
                        prob_transition = self.transitions[s2].get(s, 0)
                        prob_emission = self.emissions[s].get(observations[i], 0)
                        sum_prob += M[states.index(s2), i - 1] * prob_transition * prob_emission
                M[states.index(s), i] = sum_prob
        print(M)
        return M

    def viterbi(self, observations):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        states = list(self.transitions.keys())
        n = len(observations)

        M = np.zeros((len(states), n))
        path = np.zeros((len(states), n), dtype=int)

        for s in states:
            if s != '#':
                M[states.index(s), 0] = self.emissions[s].get(observations[0], 0) * self.transitions['#'].get(s, 0)
                path[states.index(s), 0] = 0

        for i in range(1, n):
            for s in states:
                if s != '#':
                    max_prob = 0
                    max_state = 0
                    for s2 in states:
                        if s2 != '#':
                            prob_transition = self.transitions[s2].get(s, 0)
                            prob_emission = self.emissions[s].get(observations[i], 0)
                            prob = M[states.index(s2), i - 1] * prob_transition * prob_emission
                            if prob > max_prob:
                                max_prob = prob
                                max_state = states.index(s2)
                    M[states.index(s), i] = max_prob
                    path[states.index(s), i] = max_state

        # https://stackoverflow.com/questions/17911091/append-integer-to-beginning-of-list-in-python
        best_index = np.argmax(M[:, n - 1])
        best_path = [states[best_index]]
        for t in range(n - 1, 0, -1):
            best_index = path[best_index, t]
            best_path.insert(0, states[best_index])

        print(best_path)
        return best_path


# states = ['grumpy', 'happy', 'hungry']
# observations = ['meow', 'silent', 'purr']
# transitions = {
#     '#': {'grumpy': 0.5, 'happy': 0.5},
#     'grumpy': {'grumpy': 0.3, 'happy': 0.6, 'hungry': 0.1},
#     'happy': {'happy': 0.5, 'grumpy': 0.1, 'hungry': 0.4},
#     'hungry': {'hungry': 0.3, 'grumpy': 0.6, 'happy': 0.1}
# }
# emissions = {
#     'grumpy': {'meow': 0.4, 'silent': 0.5, 'purr': 0.1},
#     'happy': {'meow': 0.3, 'silent': 0.2, 'purr': 0.5},
#     'hungry': {'meow': 0.6, 'silent': 0.2, 'purr': 0.2}
# }
# hmm = HMM(transitions, emissions)
# observation_seq = ['purr', 'silent', 'silent', 'meow', 'meow']
# forward_probabilities = hmm.forward(observation_seq)
# print(forward_probabilities)


if __name__ == "__main__":
    model_file = sys.argv[1]
    operation = sys.argv[2]

    hmm = HMM()
    hmm.load(model_file)

    if operation == "--generate":
        n = int(sys.argv[3])
        hmm.generate(n)
    elif operation == "--forward":
        observation_file = sys.argv[3]
        with open(observation_file, 'r') as f:
            observations = f.read().split()
        forward_probabilities = hmm.forward(observations)
    elif operation == "--viterbi":
        observation_file = sys.argv[3]
        with open(observation_file, 'r') as f:
            observations = f.read().split()
        most_likely_states = hmm.viterbi(observations)
