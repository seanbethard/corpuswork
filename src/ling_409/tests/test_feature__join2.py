
import nltk

"""
Load the grammar and set up the parser:
"""

grammar = nltk.data.load('file:agreement_join.fcfg',cache=False)

parser = nltk.parse.FeatureChartParser(grammar) # add trace=2 for more detailed output


"""
Specify a set of test sentences. This set can be small initially,
but should be extended for more serious testing.
"""

agreement_test_sentences = ['John and Mary left', 'John and Mary',
                            'John came and Mary left', 'Mary sang and danced',
                            'Mary baked a soft and tasty cake']

"""
Test the grammar on the given sentences:
"""

for sent in agreement_test_sentences:
    print sent + '\n'
    trees = parser.nbest_parse(sent.split())
    if len(trees) == 0:
        print '--> ungrammatical\n'
    else:
        for tree in trees:
            print tree
            print '\n'
