
from circuit_config import *
from full_pipeline import *


'''
Above are two required imports to run the training pipeline


Then go to the circuit_config.py, all our testing circuit configuration is in that python file. Choose anyone you like
Below example will use the two_stage_circuit

You can provide additional training parameter to overwrite the default training parameter

Then for simulator name choose any name you want to uniquely identify your circuit and training run

Last you can just call the CrossFoldValidationFullPipeline to run our whole pipeline
You can also provide some additional training parameter like subset, loss, model for example.
Detail parameter that can be overwrite please go check full_pipeline.py

'''

simulator = two_stage_circuit()
simulator_name = "two-stage-test"

CrossFoldValidationFullPipeline(simulator, simulator_name, {})