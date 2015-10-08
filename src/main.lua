--
-- Created by IntelliJ IDEA.
-- User: James
-- Date: 10/6/15
-- Time: 10:27 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'
require 'nn'
require 'image'

require 'math.lua'
require 'util.lua'
require 'net.lua'

-- get the dataset
if not fileExists('cifar10-test.t7') or not fileExists('cifar10-train.t7') then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
    os.execute('rm -rf cifar10torchsmall.zip')
end
trainset = torch.load('cifar10-train.t7')
trainset.data = trainset.data:double()
testset = torch.load('cifar10-test.t7')
testset.data = testset.data:double()
classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck' }

-- stochastic gradient descent requires :size() and [] indexing
setmetatable(trainset,
    {__index = function(t, i)
        return {t.data[i], t.label[i]}
    end}
);

function trainset:size()
    return self.data:size(1)
end

print(trainset:size())
print(trainset[1])
print(trainset.data:size())

-- create the network and criterion
net = buildNeuralNetwork()
print(net:__tostring());
criterion = nn.ClassNLLCriterion()

-- start passing things into the network
input = torch.rand(3,32,32)
output = net:forward(input)

-- let's say the groundtruth was class number: 3
criterion:forward(output, 3)
gradients = criterion:backward(output, 3)
gradInput = net:backward(input, gradients)
print(#gradInput)