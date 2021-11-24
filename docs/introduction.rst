EMLlib - An Empirical Model Learning Library
============================================

What is Empirical Model Learning?
---------------------------------

Empirical Model Learning (EML) is a methodology used to combine machine learning models and combinatorial systems to achieve a more
a powerful system able to solve high complexity problems.

This method was developed by Michele Lombardi, Michela Milano, and Andrea Bartolini from the University of Bologna in 2017, the paper
is available here: https://www.sciencedirect.com/science/article/pii/S0004370216000126

Combinatorial methods have a variety of applications, from the industry to decision making. These models rely on some
declarative system description. Building this model is usually a complex task since real-world systems can involve many
variables. In order to build a good model, some simplifications are introduced in the model describing the systems, this results
in some loss in accuracy.

This system can be usually described using a function that correlates some scenario with the associated decision. Since building
a hand-crafted mathematical model describing this relation is too complex, EML uses machine learning methods to approximate
this function, then encapsulates the resulting model into the optimization problem.

How to design a problem using EML
---------------------------------

In order to build an optimization model based on EML it is required to:

1. Define the core combinatorial structure of the problem
2. Obtain a machine learning model
3. Embed the empirical model in the combinatorial problem

The third step is the key to this new approach. Embed the empirical model in the optimization model requires:

* to encode the Empirical Model in terms of variables and constraints
* to define a procedure that actually improves the optimization process using the machine learning model

EMLlib is built to ease the process of integration and make EML more accessible to developers.

Applications
------------

The range of potential applications of EML is quite vast and includes:

1. Applying Combinatorial Optimization to Complex Systems (in the proper sense), or systems that are too complicated to obtain an expert-design, hand-crafted model.
2. Enabling prescriptive analytics by taking advantage of a pre-extracted predictive analytics model.
3. Enable indirect interaction between a high-level optimizer and a lower-level optimizer (whose approximate behavior can be captured via Machine Learning).
4. Based on 3, it is possible to use EML to enable multi-level optimization and therefore optimization over large-scale systems.
5. Self-adapting systems, that could be obtained by retraining the Empirical Model

So far, the method has been applied to two thermal-aware workload dispatching problems. Both problems are defined over a multi-core CPU by Intel (called SCC) featuring thermal controllers that abruptly reduce the operating frequency of each core if a threshold temperature is exceeding.

Obtaining a hand-crafted model for this system is very difficult since:

1) The temperature of each core depends on a number of complex factors
2) The platform is subject to the action of low-level controllers (besides the thermal controller, each core is managed by an online, thermal-aware, scheduler).

EML allows to extract an approximate model from data and hence to define Combinatorial Optimization problems over this platform. In particular, we define:

We are looking for more application examples! If you manage to apply EML to a new problem, we would love to hear it and post a link on this website.

EML Library
-----------

The library provides a method for applying the EML framework with Decision Trees, now compatible with scikit-learn models,
and Neural Networks, now compatible with keras models, these models will be embedded into a cplex-based problem.

For both ML approches are available the following methods:

* A reader, that creates an internal representation (optimization-friendly) of the empirical model.
* Pre and post-processing methods, i.e. bound tightening procedures
* An actual embedding technique encoding the model into the combinatorial system

One of the main focuses of our group is to make this library as flexible as possible, therefore is our goal
to extend this project to be compatible with the main machine learning libraries and solvers out there.

Structure
---------

Every object in the EMLlib belong to one of these groups:

* `Core EML objects`, the classes that implement the main EML functionalities, such as encoding, pre, and pros processing methods
* `Backends`, the object proving an interface with specific optimization solvers
* `Readers`, the classes proving a means to convert models built using popular ML framework into internal formats used by EMLlib

Based on these classes the internal structure of the library is defined as follows:

* `backend module`, which contains the submodels and methods that provide an interface with specific solvers
* `tree` and `net`, the two modules containing the classes describing the EM using an internal representation, providing the encoding functions and the readers.
* `util`, the module containing base methods used throughout all the classes in the library

The whole library is implemented using Python (>= 3.6).

Installation
------------

In order to use EMLlib you need:

* `python` (>= 3.6)  installed on your computer
* Install EMLlib typing the following command in the terminal:
    .. code-block:: python

        pip install -i https://test.pypi.org/simple/ emllib
* Then you can easily use all the library functions described previously by typing in your python file:
    .. code-block:: python

        import eml
