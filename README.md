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

Nest step is creating of weights matrices. In python it is a simple step with NumPy library and np.random.rand() function or with numpy.random.normal() for normal distributed values. In php I will first try to do it manually in a private function in my class.

For testing that all matrix values are in range between -1 and 1 I found a nice trick with array_map function:

* `min(array_map('min', $array))`
* `max(array_map('max', $array))`

After that I found an example how to implement normal distribution with Box-Muller transform on this pages:

* [PHP example for Box-Muller transform](https://www.php.net/manual/en/function.stats-rand-gen-normal.php)
* [Box-Muller transform](https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform)

As second option I will try a PHP-ML library or maybe other one, google knows different ways to create matrix in PHP.

Dot product calculation is done with three for-loops and check for right dimensions, Claude 3 Haiku LLM was a big helper here :-)

The same with sigmoid activation function, but only with one loop because I assume that only vectors are used for this calculation.

Not so easy was implementing for transpose function, from AI I get a hint with array_map function, but this doesn't work in the same way for one dimension arrays like this [1.0, 0.5, -1.5].

Now we can start with training...
