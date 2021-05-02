import sys
from models import VGG16Base,VGG16Attention
import utils
import json

def test(model_name, parameters,model_file):
    # Read the parameters
    print('-----Parameter Reading Started-----\n')
    f = open(parameters_file)
    parameters = json.load(f)
    print('-----Parameter Reading Complete-----\n')

    # Instantiate the model
    print('-----Model Instantiation Started-----\n')
    if model_name == 'VGG16':
        model = VGG16Base(num_classes=parameters['num_classes'])
    else:
        model = VGG16Attention(num_classes=parameters['num_classes'])
    
    parameters['model_descriptor'] = model_name
    parameters['model_file'] = model_file
    
    print('-----Model Instantiation Complete-----\n')

    # Perform testing
    TEST_LOSS,TEST_CLASS_AUC = utils.perform_testing(model,parameters)

    return TEST_LOSS,TEST_CLASS_AUC

if __name__ == '__main__':
    
    parameters_file = './test_params.json'
    model_name = None
    model_file = None
    
    if len(sys.argv) == 2:
        model_name = sys.argv[1]

    if len(sys.argv) == 3:
        model_name = sys.argv[1]
        model_file = sys.argv[2]

    if len(sys.argv) == 4:
        model_name = sys.argv[1]
        model_file = sys.argv[2]
        parameters_file = sys.argv[3]
        
    # Check if correct model name is specified
    if model_name in ['VGG16','VGG16-ATTN']:
      if model_file is None:
        print('Specify trained model filename')
      else:
        # Initiate Testing
        test(model_name,parameters_file,model_file)
    else:
        print('Invalid model name provided. Correct name VGG16 or VGG16-ATTN')