import nltk

grammar = nltk.data.load('file:agree_adjunct.fcfg',cache=False)

parser = nltk.parse.FeatureChartParser(grammar)

agreement_test_sentences = ['Often John left','John left often',
                            'John often left',
                            'Because John left Mary cried',
                            'Mary cried because John left',
                            'Mary because John left cried', 
                            'Through the door John left',
                            'John left through the door']

for sent in agreement_test_sentences:
    print sent + '\n'
    trees = parser.nbest_parse(sent.split())
    if len(trees) == 0:
        print '--> ungrammatical\n'
    else:
        for tree in trees:
            print tree
            print '\n'
