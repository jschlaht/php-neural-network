# how to program neural network?

for better understanding of neural networks or how neural network works it could be a good idea to program such a network by yourself. fortunately I found  a book written by Tariq Rashid about exactly this [make your own neural network](https://howtolearnmachinelearning.com/books/machine-learning-books/make-your-own-neural-network/).

In this book author use python as program language for building neural network. I will try to do the same with the programming language in wich I have the most experience, with PHP.

The simplest way for me to start with PHP project is to create one with symphony cli, I use skeleton for api project and will use unit tests.

`symfony new --dir=php-neural-network --api`

`symfony composer require symfony/maker-bundle --dev`

`symfony composer require --dev phpunit/phpunit:^9.6`

`symfony composer require --dev symfony/test-pack`

`symfony console make:test`


First step is to build a neuralNetwork class as a service in app/Service. This class has a constructor and two methods: train and query.

Next step is to initialise network with input parameters in constructor. We have different numbers for input, hidden and output nodes and a value for learning rate. I set this in constructor and created getter methods.

Nest step is creating of weights matrixes. In python it is a simple step with NumPy library and np.random.rand() function. In php I will first try to do it manually in a private function in my class.

As second option I will try a PHP-ML library or maybe other one, google knows different ways to create matrix in PHP.
