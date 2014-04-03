# Spearmint Salad
This python software is an implementation of the Sequential Model-Based Ensemble Optimization (ESMBO) algorithm [[1](http://arxiv.org/abs/1402.0796)].  It combines [Spearmint](https://github.com/JasperSnoek/spearmint) [[2](https://nips.cc/Conferences/2012/Program/event.php?ID=3571)] for fast hyperparameter optimization with the Agnostic Bayes theory [[3](http://jmlr.org/proceedings/papers/v32/lacoste14.html)] to generate an ensemble of learning algorithms over the hyperparameter space for increasing the generalization performances.

## Features
* **Fast hyperparameter optimization** via [Spearmint](https://github.com/JasperSnoek/spearmint), based on a gaussian process modelization of the hyperparameter search space.
* **Ensemble of learning algorithms** with no extra computational cost at learning time. 
* **3D vizualization** of the hyperparameter space based on [mayavi](http://code.enthought.com/projects/mayavi/) (Optional dependency).
* **Easy parallelization** via python's multiprocessing or an implementation of an mpi queue for running on a computer grid.
* **Crash recovery** (coming soon). 
* **Anytime algorithm.** You can obtain the best predictor so far and visualize the behavior at anytime.
* **MongoDB compatible**. All information gathered during the optimization process is stored in a MongoDB-like structure. 

## Dependencies
* numpy, scipy
* matplotlib (Optional, for visualization)
* mayavi, traits (Optional, for 3D visualization)
* mpi4py, openmpi (Optional, for parallelization over computational grid)
* scikit-learn (Optional, for playing with the provided examples)

## Usage
```
from spearmint_salad import  hp
from sklearn.svm import SVC
hp_space = hp.Obj(SVC)(
    C = hp.Float( 0.01, 1000, hp.log_scale ),
    gamma = hp.Float( 10**-5, 1000, hp.log_scale ),
)
```

## References
[1] Lacoste, Alexandre, Hugo Larochelle, François Laviolette, and Mario Marchand. "Sequential Model-Based Ensemble Optimization." arXiv preprint arXiv:1402.0796 (2014).

[2] Snoek, Jasper, Hugo Larochelle, and Ryan P. Adams. "Practical Bayesian Optimization of Machine Learning Algorithms." In NIPS, pp. 2960-2968. 2012.

[3] Lacoste, Alexandre, Mario Marchand, François Laviolette, and Hugo Larochelle. "Agnostic Bayesian Learning of Ensembles." In Proceedings of The 31st International Conference on Machine Learning, pp. 611-619. 2014.

