from mnist import MNIST
mndata = MNIST('/Users/mine/Downloads/')
mndata.load_training()
mndata.load_testing()

print(mndata.load_training())