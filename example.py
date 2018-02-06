import src as lightneuralnetwork
#if outside of project directory replace above with
#import lightneuralnetwork


if __name__ == "__main__":
    example_inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    example_outputs = [
        [0],
        [0],
        [0],
        [1]
    ]

    #this is the format for adding training examples
    example_set = [example_inputs, example_outputs]

    #training examples can be passed through initialization or set later
    net = lightneuralnetwork.Net([2, 12, 1], example_set)

    #initial weights are random by default, so this output will be random
    print net.run(example_inputs)
    #train the network
    net.train()
    #after training running the net should be close to example outputs
    #if the net was able to find appropriate weights
    print net.run(example_inputs)
    #compare the resulting matrix to example outputs

