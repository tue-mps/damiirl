# DAMIIRL

This repository is the implimention the paper:

**Deep Adaptive Multi-Intention Inverse Reinforcement Learning</a>**
<br>
<a href="http://web.stanford.edu/~agrim/">Ariyan Bighashdel</a>,
<a href="https://www.tue.nl/en/research/researchers/panagiotis-meletis/">Panagiotis Meletis</a>,
<a href="https://www.tue.nl/en/research/researchers/pavol-jancura/">Pavol Jancura</a>,
<a href="https://www.tue.nl/en/research/researchers/gijs-dubbelman/">Gijs Dubbelman</a>
<br>
Accepted for presentation at [ECML PKDD 2021](https://2021.ecmlpkdd.org/)

In this paper, two algorithms, namely "SEM-MIIRL" and "MCEM-MIIRL" are developed which can learn an a priori unknown number of nonlinear reward functions from unlabeled experts' demonstrations. The algorithms are evaluated on two proposed environments, namely "M-ObjectWorld" and "M-BinaryWorld". The proposed algorithms, and the environments are implemented in this repository.

If you find this code useful in your research then please cite
```
@inproceedings{gupta2018social,
  title={Deep Adaptive Multi-Intention Inverse Reinforcement Learning},
  author={Bighashdel, Ariyan and Meletis, Panagiotis and Jancura, Pavol and Dubbelman, Gijs},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  number={CONF},
  year={2018}
}
```

## Dependencies
The code is developed and tested on Ubuntu 18.04 with Python 3.6 and PyTorch 1.9.

You can install the dependencies by running:

```bash
pip install -r requirements.txt   # Install dependencies
```

Implimentation of "Deep Adaptive Multi-intention Inverse Reinforcement Learning"

## Training

Dependencies:

Python = 3.6
Torch = 1.7
numpy = 1.19

Usage:
run python3 main.py for defualt training of SEM-MIIRL on M-BinaryWorld with three rewards types.
