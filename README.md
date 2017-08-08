# pytorch-nin-cifar10
pytorch implementation of network-in-network model on cifar10. All settings are base on the [network-in-network model in Caffe model zoo](https://gist.github.com/mavenlin/e56253735ef32c3c296d).

## Instructions
```bash
$ git clone https://github.com/jiecaoyu/pytorch-nin-cifar10.git
$ cd pytorch-nin-cifar10
$ mkdir data
```
Then download the data from this link and uncompress it into the ```./data/``` directory. Now you can train the model by running
```bash
$ python original.py
```

## Accuracy
By tweaking hyper-parameters, the model can reach the accuracy of 89.64%, which is better than other available implementations.

## License
The data used to train this model comes from http://www.cs.toronto.edu/~kriz/cifar.html Please follow the license there if used.
