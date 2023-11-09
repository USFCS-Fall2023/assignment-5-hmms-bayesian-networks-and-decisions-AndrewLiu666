import HMM
import alarm
import carnet


def main():
    hmm1 = HMM.HMM()
    hmm1.load('two_english')
    hmm2 = HMM.HMM()
    hmm2.load('partofspeech.browntags.trained')
    hmm2.generate(20)
    with open("ambiguous_sents.obs", 'r') as f:
        observations = f.read().split()
        hmm2.forward(observations)
        hmm2.viterbi(observations)
    alarm.main()
    carnet.main()


main()
