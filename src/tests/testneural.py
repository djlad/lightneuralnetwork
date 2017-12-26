import unittest
import neural as net_module

class TestNet(unittest.TestCase):
    def test_and_gate_error(self):
        layerSizes = [2,3,1]
        inputs = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]
        outputs = [[0],[0],[0],[1]]
        training_set = [inputs, outputs]
        net = self.sample_net_1()
        net.training_set = training_set
        cost = net.calc_cost()
        expected_output = [ 0.490794321114148, 0.55743360861882951,
                            0.54113913657798884, 0.050904261638902749]
        for ao, eo in zip(cost, expected_output):
            self.assertEquals(ao[0, 0], eo)
    
    def test_set_weight(self):
        net = net_module.Net([2,3,1])
        net.set_weight(0, 0, 0, 5)
        net.set_weight(0, 0, 1, 6)
        net.set_weight(1, 0, 1, 7)
        net.set_weight(1, 0, 2, 8)
        self.assertEquals(net.layers[0].weights[0,0], 5)
        self.assertEquals(net.layers[0].weights[1,0], 6)
        self.assertEquals(net.layers[1].weights[1,0], 7)
        self.assertEquals(net.layers[1].weights[2,0], 8)
    
    def test_run(self):
        net = self.sample_net_1()
        test_input = [1,1]
        test_output = net.run(test_input)
        self.assertEquals(test_output[0,0], 0.7743802720529458)
    
    #generated test data:
    def sample_net_1(self):
        layerSizes = [2, 3, 1]
        net = net_module.Net(layerSizes)
        #weights for layer 0:
        net.set_weight(0, 0, 0, .8)
        net.set_weight(0, 0, 1, .2)
        net.set_weight(0, 1, 0, .4)
        net.set_weight(0, 1, 1, .9)
        net.set_weight(0, 2, 0, .3)
        net.set_weight(0, 2, 1, .5)
        #weights for layer 1:
        net.set_weight(1, 0, 0, .3)
        net.set_weight(1, 0, 1, .5)
        net.set_weight(1, 0, 2, .9)
        return net
        

if __name__ == "__main__":
    unittest.main()