from network.TwoLayerNet import TwoLayerNet

if __name__ == '__main__':
    net = TwoLayerNet(input_size=10, hidden_size=10, output_size=10)
    print(net.params)
