caltechx.go
===========

The is a [golang](http://golang.org) version of the Machine learning course from caltech: [Learning from data](https://work.caltech.edu/telecourse.html).

* **week 1:**
    * PLA (Perceptron learning Algorithm)
* **week 2:**
    * Hoeffding Inequality
    * Linear Regression
    * Nonlinear Transformation
* **week 3:**
    * Generalization Error
* **week 4:**
    * VC bound
    * Bias and Variance
* **week 5:**
    * Linear Regression Error
    * Gradient Descent
    * Logistic Regression
* **week 6:**
    * Overfitting and Regularization With Weight Decay
    * Neural Networks
* **week 7:**
    * Validation
    * Estimators
    * Cross Validation
    * PLA vs. SVM
* **week 8:**
    * Support Vector Machines With Soft Margins
    * Polynomial Kernels
    * Cross Validation
    * RBF Kernel

##Build:
There is a specific file `week<x>.go` for the work of each week (1 to 8). To build it run the following command where `x = 1`

    go build week1.go
##Run:
Similarly you can run the work of a specific week as follows:

    ./week1


##Todo:
* refactor
* concurrent runs.
* command line animations. [Pretty command line / console output on Unix in Python and Go Lang](http://www.darkcoding.net/software/pretty-command-line-console-output-on-unix-in-python-and-go-lang/)
* refactor PLA and other functions into separate packages.
* linear regression should have a Xn array and an Zn collection when a transformation takes place
* add transpose function.
* transformation function should accept array with param x0 = 1 to transform
* better and consistent print statements.
* catch all error and have all functions send errors.
* add tests

##Thoughts:

It might be better to divide the packages based on *models* and *methods*.
Here is how the topics are presented in the learning from data web page:
[topics](http://work.caltech.edu/library/)

###models:

* linear classification: PLA
* linear regression
* logistic regression
* non linear transformation
* neural networks
* support vector machines
* nearest neighbors


###methods:
* regularization
* validation
