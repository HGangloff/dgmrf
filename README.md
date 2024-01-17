dgmrf
=====

Deep Gaussian Markov Random Fields and their extensions. Non-official reimplementations of the following models (work in progress):

```
@inproceedings{siden2020deep,
  title={Deep gaussian markov random fields},
  author={Sid{\'e}n, Per and Lindsten, Fredrik},
  booktitle={International conference on machine learning},
  pages={8916--8926},
  year={2020},
  organization={PMLR}
}

@inproceedings{graph_dgmrf,
    author = {Oskarsson, Joel and Sid{\'e}n, Per and Lindsten, Fredrik},
    title = {Scalable Deep {G}aussian {M}arkov Random Fields for General Graphs},
    booktitle = {Proceedings of the 39th International Conference on Machine Learning},
    year = {2022}
}

@article{lippert2023deep,
  title={Deep Gaussian Markov Random Fields for Graph-Structured Dynamical Systems},
  author={Lippert, Fiona and Kranstauber, Bart and van Loon, E Emiel and Forr{\'e}, Patrick},
  journal={arXiv preprint arXiv:2306.08445},
  year={2023}
}
```

# Installation

Install the latest version with pip

```bash
pip install dgmrf
```

# Documentation

The project's documentation is available at [https://hugo.gangloff.gitlab.io/dmgrf/index.html](https://hugo.gangloff.gitlab.io/dgmrf/index.html)

# Contributing

* First fork the library on Gitlab.

* Then clone and install the library in development mode with

```bash
pip install -e .
```

* Install pre-commit and run it.

```bash
pip install pre-commit
pre-commit install
```

* Open a merge request once you are done with your changes.

# Contributors

*Active*: Hugo Gangloff
