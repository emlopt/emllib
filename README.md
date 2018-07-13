# EMLlib - An Empirical Model Learning Library

This repository contains an implementation of techniques related to [Empirical Model Learning](http://emlopt.github.io), a technique to enable Combinatorial Optimization and decision making over complex real-world systems. The method is based on the (relatively simple) idea of:

1. Using a Machine Learning (ML) model to approximate the input/output behavior of a system that is hard to model by conventional means.
2. Embedding such _Empirical Model_ into a Combinatorial Optimization model.

The emphasis of EML is mostly on the techniques to perform the embedding. These should be designed so that the optimization engine can exploit the structure of the empirical model to boost the search process.

This repository includes:

* Actual embedding techniques (encodings and global constraints)
* Pre- and post- processing methods
* I/O support (in particular readers for popular ML libraries)

The EML technique has been originally designed by [the Artificial Intelligence group of the University of Bologna](http://ai.unibo.it), but the repository implements also contributions from other groups.

The repository has been just launched and currently contains only a fraction of the existing techniques related to EML. In particular, we have published the bare minimum to support the EML tutorial at IJCAI 2018, and the current documentation is very scarce. More content (and improved documentation) will be released in the coming months.

The EMLlib is part of the EML resources (including the EML tutorial), all available at [http://emlopt.github.io](http://emlopt.github.io).

## Structure and Installation

In details, every object in the EMLlib belongs to one of three groups:

* *Core EML objects* are the classes and functions that implement the main EML functionalities, such as encodings, pre- and post-processing methods
* *Backends* provide the interface the core EML objects with a specific optimization solver
* *Readers* allow to convert Machine Learning models built via popular frameworks in the internal formats used by the EMLlib

The repository is structured as follows:

* The `backend` module contains all the available backends
  * The `base` submodule defines the interface that all backends should implement
  * The remaining submodules provide solver-specifc implementations of the base API
* The `net` and `tree` modules contains classes and functions to work respectively with Machhine Learning models, and in particular Neural Network and Decision Trees
  * The `describe` submodule contains classes and functions that are useful to *represent* ML models
  * The `embed` submodule contains classes and functions that are useful to *embed* ML models into a combinatorial model
  * The `process` submodule contains classes and functions that are useful to *pre- and post-process* ML models for use optimization
  * The `reader` submodule contains specific reader implementations
* The `util` module contains classes and functions used by multiple core EML objects

The core EML library is implemented in pure Python (3.X) and relies on very few additional modules. However, for every practical task you will need to use at least one backend, and probably at least one reader. 

### Installing the Core EML API

In order to have the core EML functionalities working you will need to:

* Obtain a Python 3.X distribution
* Install numpy, e.g. via pip with `pip3 install numpy`
* Add the main folder of your repository to your python path. This can be done by setting the global `PYTHONPATH` variables to include the path to this repository

We are planning to add a pip based installation in the coming months.

### Installing the Keras Network Reader

A reader for Neural Networks built with the [keras sequential API](???) is provided in `net.reader.keras_reader`.

The reader itself does not require a specific installation, but you will need to install [keras](???) to make it work.

### Installing the Sklearn Decision Tree Reader

A reader for Decision Trees built with the [scikit-learn](???) is provided in `tree.reader.sklearn_reader`.

The reader itself does not require a specific installation, but you will need to install [scikit-learn](???) to make it work.


### Installing the Cplex Backend

A backend for the [CPLEX Mixed Integer Linear Solver]() is provided in `backend.cplex_backend`.

The backend itself does not require a specific installation, but you will need to install [docplex](???) to make it work.

