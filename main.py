
from pipeline import pipeline
from utils import *



if __name__ == '__main__':
    multiple_config = load_yaml(os.path.join(CONFIG_PATH,'multiple_pipeline.yaml'))
    for circuit in multiple_config['circuits']:
        print('Working on circuit: ', circuit)
        pipeline(circuit)

