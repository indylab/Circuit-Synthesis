# Circuit-Synthesis



### Circuit Synthesis is a project try to use machine learning to generate circuit design giving required properties


1. Running Problem 1: base dataset, DL, across data sizes with base test dataset, success rate as function of (two-sided) error margin:  python main.py --path=./config/config_template/problem1-compare-datasize-relative-Error-margin.yaml
2. Running Problem 2 Number 1: Compare datasets with DL, 10-fold CV as a function of error margin:  python main.py --path=./config/config_template/problem2-compare-dataset-DL-10fold-absolute-Error-margin.yaml
3. Running Problem 2 Number 2: Test success rate; Compare training methods (DL, lookup, RF, …) with “softargmax”, 10-fold CV as a function of error margin: python main.py --path=./config/config_template/problem2-compare-method-Softargmax-10fold-absolute-Error-margin.yaml
4. Running Problem 2 Number 3: Test success rate; Compare data sizes with DL, “softargmax” as a function of error margin:  python main.py --path=./config/config_template/problem2-compare-datasize-softArgmax-DL-Absolute-Error-margin.yaml