
require 'torch'
require 'nn'

local function RyaNet(ds)
    -- sequential network
    local net = nn.Sequential()

    -- 3 input channels, 7 output channels, 5x5 kernel, 1x1 stride, tanh, 2x2 pool, 2x2 stride
    net:add(nn.SpatialConvolution(3, 7, 5, 5, 1, 1))
    net:add(nn.Tanh())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- 7 input channels, 14 output channels, 5x5 kernel, 1x1 stride, tanh, 2x2 pool, 2x2 stride
    net:add(nn.SpatialConvolution(7, 14, 5, 5, 1, 1))
    net:add(nn.Tanh())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- 14 input channels, 21 output channels, 5x5 kernel, 1x1 stride, tanh, 2x2 pool, 2x2 stride
    net:add(nn.SpatialConvolution(14, 21, 5, 5, 1, 1))
    net:add(nn.Tanh())
    net:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- 21 input channels, 50 output channels, 2x2 kernel, 1x1 stride, tanh, 3x3 pool, 3x3 stride
    net:add(nn.SpatialConvolution(21, 50, 2, 2, 1, 1))
    net:add(nn.Tanh())
    net:add(nn.SpatialMaxPooling(3, 3, 3, 3))

    -- prepare for fully connected layers
    net:insert(nn.Convert(ds:ioShapes(), 'bchw'), 1)
    net:add(nn.Collapse(3))

    -- fully connected layer, 5 input channels, 3 output channels (classes)
    net:add(nn.Linear(50, 5))
    net:add(nn.Linear(5, 2))

    -- converts the output to a log-probability, useful for classification
    net:add(nn.LogSoftMax())

    return net
end

local function pick_model(arg, ds)
    local model = nil
    if arg == 'RyaNet' then
        model = RyaNet
    else
        assert(false)
    end

    return model(ds)
end


return pick_model
