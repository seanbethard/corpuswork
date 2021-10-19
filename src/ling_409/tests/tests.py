"""

import nltk

"""
Load the grammar and set up the parser:
"""

grammar = nltk.data.load('file:and_2.fcfg',cache=False)

parser = nltk.parse.FeatureChartParser(grammar) # add trace=2 for more detailed output


"""
Specify a set of test sentences. This set can be small initially,
but should be extended for more serious testing.
"""

agreement_test_sentences = ['Mary sang and danced',
                            'Mary and John danced',
                            'Mary and John',
                            'Mary and John saw Kim',
                            'Mary saw John and Kim',
                            'Mary sang and John danced',
                            'Mary baked a soft and tasty cake',
                            'Mary baked a tasty quiche and a soft cake']

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

