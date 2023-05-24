# Learning to Design Analog Circuits to Meet Threshold Specifications



## Overview
 Code for Learning to Design Analog Circuits to Meet Threshold Specifications (Accepted by ICML 2023 as Poster)

 - Preprint Coming Soon
 - Website Comning Soon

## Docker
To export running results to host machine: 

1. Create two folders, one for out_plot and another for result_out. 
2. Run the docker using command 

```
docker run -v {absolute path to out_plot folder}:/Circuit-Synthesis/out_plot -v {absolute path to result_out folder}:/Circuit-Synthesis/result_out {docker image name} --path={Train config path}
```

## Usage
### Problem 1 
Base dataset, DL, across data sizes with base test dataset, success rate as function of (two-sided) error margin:  
```
python main.py --path=./config/config_template/problem1-compare-datasize-relative-Error-margin.yaml
```
### Problem 2 Number 1 and Number 4
Compare datasets construction methods using deep learning, 10-fold Cross Validation as a function of error margin:  
```
python main.py --path=./config/config_template/problem2-compare-dataset-DL-10fold-absolute-Error-margin.yaml
```

### Problem 2 Number 2
Test success rate; Compare training methods (DL, lookup, RF, …) with “softargmax”, 10-fold cross validation as a function of error margin: 
```
python main.py --path=./config/config_template/problem2-compare-method-Softargmax-10fold-absolute-Error-margin.yaml
```

### Problem 2 Number 3
Test success rate; Compare data sizes with DL, “softargmax” as a function of error margin:  

```
python main.py --path=./config/config_template/problem2-compare-datasize-softArgmax-DL-Absolute-Error-margin.yaml
```


## Citation
(Coming soon)
