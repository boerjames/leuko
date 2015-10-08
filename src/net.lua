--
-- Created by IntelliJ IDEA.
-- User: James
-- Date: 10/8/15
-- Time: 10:36 AM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'nn'

function buildNeuralNetwork()
    -- sequential network
    local net = nn.Sequential()

    -- input image 3 channel, 6 output channels, 5x5 convolution kernel
    net:add(nn.SpatialConvolution(3, 6, 5, 5))

    -- A max-pooling operation that looks at 2x2 windows and finds the max.
    net:add(nn.SpatialMaxPooling(2,2,2,2))

    net:add(nn.SpatialConvolution(6, 16, 5, 5))
    net:add(nn.SpatialMaxPooling(2,2,2,2))

    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
    net:add(nn.View(16*5*5))

    -- fully connected layer (matrix multiplication between input and weights)
    net:add(nn.Linear(16*5*5, 120))
    net:add(nn.Linear(120, 84))

    -- 10 is the number of outputs of the network (in this case, 10 digits)
    net:add(nn.Linear(84, 10))

    -- converts the output to a log-probability. Useful for classification problems
    net:add(nn.LogSoftMax())

    return net
end

