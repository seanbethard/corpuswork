import nltk

"""
Load the grammar and set up the parser:
"""

grammar = nltk.data.load('file:agreement.fcfg',cache=False)

parser = nltk.parse.FeatureChartParser(grammar) # add trace=2 for more detailed output


"""
Specify a set of test sentences. This set can be small initially,
but should be extended for more serious testing.
"""

agreement_test_sentences = ['I likes Jody','you likes Jody',
                            'he likes Jody','the dog likes Jody',
                            'we likes Jody','they likes Jody',
                            'children likes Jody',
							'John not might not have been not eating not apples',
							'John not has not been not eating not apples',
							'John not is not eating not apples',
							'John not eats not apples']

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

