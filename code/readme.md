# Code for Learned ISTA with Error-Based Thresholding for Adaptive Sparse Coding

The paper can be found [here](https://ieeexplore.ieee.org/abstract/document/10446361).

This repository is implemented on the base of [ALISTA](https://github.com/VITA-Group/ALISTA).


## Execution

### Baseline: LISTA_cp
```
python main.py --net LISTA_cp --SNR inf --gpu 0 -M 250 -N 500 -u -fixval False -id 0 -P 0.1 (-t)
```

### Baseline: LISTA_cpss
```
python main.py --net LISTA_cpss --SNR inf --gpu 0 -M 250 -N 500 -u -fixval False -p 1.2 -maxp 13 -id 0 -P 0.1 (-t)
```

### Baseline: ALISTA
```
python main.py --net ALISTA --SNR inf --gpu 0 -M 250 -N 500 -u -fixval False -p 1.2 -maxp 13 -id 0 -P 0.1 -W data/W.npy (-t)
```

### EBT-LISTA_cp
```
python main.py --net LISTA_cp --SNR inf --gpu 0 -M 250 -N 500 -u -fixval False -id 0 -P 0.1  -ad -bias (-t)
```

### EBT-LISTA_cpss
```
python main.py --net LISTA_cpss --SNR inf --gpu 0 -M 250 -N 500 -u -fixval False -p 1.2 -maxp 13 -id 0 -P 0.1  -ad -bias (-t)
```

### EBT-ALISTA
```
python main.py --net ALISTA --SNR inf --gpu 0 -M 250 -N 500 -u -fixval False -p 1.2 -maxp 13 -id 0 -P 0.1 -W data/W.npy  -ad -bias (-t)
```
