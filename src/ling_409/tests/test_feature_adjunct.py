
import nltk

"""
Load the grammar and set up the parser:
"""

grammar = nltk.data.load('file:agreement_adjunct.fcfg',cache=False)

parser = nltk.parse.FeatureChartParser(grammar) # add trace=2 for more detailed output


"""
Specify a set of test sentences. This set can be small initially,
but should be extended for more serious testing.
"""

agreement_test_sentences = ['Often John left','John left often',
                            'John often left',
                            'Because John left Mary cried',
                            'Mary cried because John left',
                            'Mary because John left cried', 
                            'Through the door John left',
                            'John left through the door']

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

