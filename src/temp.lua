require 'torch'
require 'optim'
require 'nn'
require 'xlua'
require 'paths'

if not paths.dirp('mnist.t7') then
    -- Fetch MNIST Dataset
    os.execute("curl http://torch7.s3-website-us-east-1.amazonaws.com/data/mnist.t7.tgz > mnist.t7.tgz")
    os.execute("tar xf mnist.t7.tgz")
end

train_file = 'mnist.t7/train_32x32.t7'
test_file = 'mnist.t7/test_32x32.t7'

raw_train_data = torch.load(train_file, 'ascii')
raw_test_data = torch.load(test_file, 'ascii')

-- extract 28x28 patch and flatten to raster order
raster_size = 28 * 28
-- we have 10 possible digits
class_size = 10
local function transform_mnist_data(mnist_data, index)
    -- crop inner 28x28 patch and flatten
    local flat_pixels = mnist_data.data[index][1]:sub(3, 30, 3, 30):resize(raster_size)
    local binarized_pixels = flat_pixels:apply(function (x) return x > 127; end)
    local label_class = mnist_data.labels[index]
    return binarized_pixels, label_class
end

train_count = raw_train_data.data:size()[1]
train_data = torch.Tensor(train_count, raster_size)
train_label = torch.Tensor(train_count)
for i = 1, train_count do
    train_data[i], train_label[i] = transform_mnist_data(raw_train_data, i)
end
-- reduce memory consumption by explicitly garbage collecting after a loop where we construct many variables
collectgarbage()

test_count = raw_test_data.data:size()[1]
test_data = torch.Tensor(test_count, raster_size)
test_label = torch.Tensor(test_count)
for i = 1,test_count do
    test_data[i], test_label[i] = transform_mnist_data(raw_test_data, i)
end

collectgarbage()

print(train_count)
print(test_count)

-- Initialize our network to simply have 3 Linear units.
-- References:
--  * https://github.com/torch/nn/blob/master/doc/simple.md#nn.Linear
hidden_neuron_count = 100
input_layer = nn.Linear(raster_size, hidden_neuron_count)
hidden_layer = nn.Linear(hidden_neuron_count, hidden_neuron_count)
output_layer = nn.Linear(hidden_neuron_count, class_size)

-- Initialize matrices to have gaussian distributions centered around 0.0 and standard deviation 0.001
std = 0.001
input_layer.weight = torch.randn(input_layer.weight:size(1), input_layer.weight:size(2)) * std
output_layer.weight = torch.randn(output_layer.weight:size(1), output_layer.weight:size(2)) * std

-- Initialize the hidden layer to have the identity matrix.
hidden_layer.weight = torch.eye(hidden_layer.weight:size(1), hidden_layer.weight:size(2))

-- Sequential provides a means to plug layers together in a feed-forward fully connected manner.
-- References:
--  * https://github.com/torch/nn/blob/master/doc/containers.md#nn.Sequential
model = nn.Sequential()
model:add(input_layer)
model:add(nn.ReLU())
model:add(hidden_layer)
model:add(output_layer)

function measure_performance(model, criterion, inputs, labels)
    local loss = 0
    local count = inputs:size(1)
    local correct_class = 0
    for i = 1, count do
        local predicted = model:forward(inputs[i])
        local actual_class = labels[i]
        -- get the maximum scoring class index
        local score, class_indices = torch.max(predicted, 1)
        -- this is just a 1x1 tensor, so extract the only value
        local predicted_class = class_indices[1]
        if actual_class == predicted_class then
            correct_class = correct_class + 1
        end
        loss = loss + criterion:forward(predicted, actual_class)
    end
    loss = loss / count
    local accuracy = correct_class / count
    return accuracy, loss
end

-- Measure loss via CrossEntropy.
-- References:
--  * https://github.com/torch/nn/blob/master/doc/criterion.md#nn.CrossEntropyCriterion
criterion = nn.CrossEntropyCriterion()

-- Initialize parameters for Stochastic Gradient Descent
-- References:
--  * https://github.com/torch/optim/blob/master/sgd.lua
sgd_params = {
    learningRate = 0.01,
    learningRateDecay = 0,
    weightDecay = 0.001,
    momentum = 0.9
}

best_train_accuracy = 0
best_train_accuracy_epoch = 0
best_test_accuracy = 0
best_test_accuracy_epoch = 0

-- minibatch size
batch_size = 20
epoch_count = torch.floor((train_count - 1) / batch_size)
for i = 1, epoch_count do
    xlua.progress(i, epoch_count)
    -- Get parameters and gradient for our model.
    flattened_weights, dl_dx = model:getParameters()
    -- inputs is a batch_size x 28 * 28 tensor (matrix)
    local inputs = train_data:sub(i*batch_size, (i+1) * batch_size-1)
    -- targets is a batch_size tensor (vector)
    local targets = train_label:sub(i*batch_size, (i+1) * batch_size-1)
    local lossFunction = function(x)
        -- Set weights to whatever the optimized step evaulation is. This is needed for minibatch.
        flattened_weights:copy(x)

        -- reset model's internal gradient
        dl_dx:zero()

        -- loss is the average of all items in minibatch
        local loss = 0
        for j = 1, batch_size do
            -- run the forward pass
            local output = model:forward(inputs[j])
            local err = criterion:forward(output, targets[j])
            loss = loss + err

            -- run backprop
            local dl_dt = criterion:backward(output, targets[j])
            model:backward(inputs[j], dl_dt)
        end

        dl_dx:div(batch_size)
        loss = loss / batch_size

        return loss, dl_dx
    end
    -- Use Stochastic Gradient Descent
    -- References:
    --  * https://github.com/torch/optim
    w_prime, losses = optim.sgd(lossFunction, flattened_weights, sgd_params)
    if i % 100 == 0 then
        local train_accuracy, train_loss = measure_performance(model, criterion, train_data, train_label)
        local test_accuracy, test_loss = measure_performance(model, criterion, test_data, test_label)

        if (train_accuracy > best_train_accuracy) then
            best_train_accuracy = train_accuracy
            best_train_accuracy_epoch = i
        end
        if (test_accuracy > best_test_accuracy) then
            best_test_accuracy = test_accuracy
            best_test_accuracy_epoch = i
        end

        print(string.format("Train accuracy: %s loss: %s", train_accuracy, train_loss))
        print(string.format("Test accuracy: %s loss: %s", test_accuracy, test_loss))
        collectgarbage()
    end
end

print("Best training accuracy: " .. tostring(best_train_accuracy) .. " at epoch " .. tostring(best_train_accuracy_epoch))
print("Best test accuracy: " .. tostring(best_test_accuracy) .. " at epoch " .. tostring(best_test_accuracy_epoch))