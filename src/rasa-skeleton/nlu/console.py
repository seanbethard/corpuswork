"""

create nlu interpreter
parse incoming queries

required arguments:
  model
  query

$ python console.py play-jan22 "play beyonce"

"""


import argparse
import os
from rasa_nlu.model import Interpreter

def get_test_result(nlu_result):
    """
    get test elements from nlu result
    """
    result = {}
    result['entities'] = []
    result['intent'] = nlu_result['intent']['name']
    for d in nlu_result['entities']:
        result['entities'].append({'entity':d['entity'], 'value':d['value']})
        
    return result
        


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="name of nlu model")
    parser.add_argument("query", help="input query")
    args = parser.parse_args()
    
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(dir_path, "models", "nlu", "default", args.model)
    interpreter = Interpreter.load(model_dir)
    result = interpreter.parse(args.query)
    
    print()
    print("NLU result:")
    print(result)
    
    print()
    print("Test result:")
    # TODO: csv formatting
    print("\""+args.query+"\""+','+"\""+str(get_test_result(result))+"\"")
    