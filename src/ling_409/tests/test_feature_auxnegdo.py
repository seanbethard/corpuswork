import nltk

"""
Load the grammar and set up the parser:
"""

grammar = nltk.data.load('file:agreement_auxnegdo.fcfg',cache=False)

parser = nltk.parse.FeatureChartParser(grammar) # add trace=2 for more detailed output


"""
Specify a set of test sentences. This set can be small initially,
but should be extended for more serious testing.
"""

agreement_test_sentences = ['John might eat apples',
                            'John might eat not apples',
                            'John might have eaten apples',
                            'John might eaten have apples',
                            'John might not have eaten apples',
                            'John might not have not eaten apples',
                            'John might not have eaten not apples',
                            'John might not have not been not eating apples',
                            'John might not have not been not eating not apples',
                            'John might have not eaten apples',
                            'John might have eaten not apples',
                            'John might be eating apples',
                            'John might eating be apples',
                            'John might not be eating apples',
                            'John might be not eating apples',
                            'John might be eating not apples',
                            'John eats apples',
                            'John has eaten apples',
                            'John is eating apples',
                            'John does not eat apples',
                            'John not does eat apples',
                            'John eat not apples',
                            'John has not eaten apples',
                            'John is not eating apples',
                            'John has eaten not apples',
                            'John is eating not apples']
        

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

