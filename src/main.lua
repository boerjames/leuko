require 'dp'
require 'torchx'
require 'data.lua'
require 'net.lua'

--[[how to use]]-- $> th main.lua [flag] [parameter]

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()

--[[data parameters]]--
cmd:option('--validRatio', 0.15, 'ratio to use for validation')
cmd:option('--dataPath', '/home/boer/data/jpg', 'Where to look for the images')
cmd:option('--dataSize', '{3,64,64}', 'How big the images should be')
cmd:option('--resultsPath', '/home/boer/save', 'Where to store results')

--[[network layers]]--
cmd:option('--channelSize', '{7,14,21,50}', 'Number of output channels for each convolution layer.')
cmd:option('--kernelSize', '{5,5,5,2}', 'kernel size of each convolution layer. Height = Width')
cmd:option('--kernelStride', '{1,1,1,1}', 'kernel stride of each convolution layer. Height = Width')
cmd:option('--poolSize', '{2,2,2,3}', 'size of the max pooling of each convolution layer. Height = Width')
cmd:option('--poolStride', '{2,2,2,3}', 'stride of the max pooling of each convolution layer. Height = Width')
cmd:option('--hiddenSize', '{5}', 'size of the dense hidden layers after the convolution')
cmd:option('--padding', false, 'add math.floor(kernelSize/2) padding to the input of each convolution')
cmd:option('--dropout', false, 'use dropout')
cmd:option('--dropoutProb', '{0.2,0.5,0.5}', 'dropout probabilities')

--[[network parameters]]--
cmd:option('--learningRate', 0.1, 'learning rate at t=0')
cmd:option('--lrDecay', 'linear', 'type of learning rate decay : adaptive | linear | schedule | none')
cmd:option('--schedule', '{}', 'learning rate schedule')
cmd:option('--minLR', 0.00001, 'minimum learning rate')
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')
cmd:option('--momentum', 0, 'momentum')
cmd:option('--maxEpoch', 10, 'maximum number of epochs to run')
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--batchSize', 10, 'number of examples per batch')
cmd:option('--accUpdate', false, 'accumulate gradients inplace')

--[[cuda settings]]--
cmd:option('--cuda', false, 'use CUDA')
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')

--[[preprocessing]]--
cmd:option('--standardize', false, 'apply Standardize preprocessing')
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')
cmd:option('--lecunlcn', false, 'apply Yann LeCun Local Contrast Normalization (recommended)')
cmd:option('--activation', 'Tanh', 'transfer function like ReLU, Tanh, Sigmoid')
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')

--[[verbosity]]--
cmd:option('--progress', false, 'print progress bar')
cmd:option('--silent', false, 'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
    table.print(opt)
end

opt.dataSize = table.fromString(opt.dataSize)
opt.channelSize = table.fromString(opt.channelSize)
opt.kernelSize = table.fromString(opt.kernelSize)
opt.kernelStride = table.fromString(opt.kernelStride)
opt.poolSize = table.fromString(opt.poolSize)
opt.poolStride = table.fromString(opt.poolStride)
opt.dropoutProb = table.fromString(opt.dropoutProb)
opt.hiddenSize = table.fromString(opt.hiddenSize)

--[[preprocessing]]--
local input_preprocess = {}
if opt.standardize then
    table.insert(input_preprocess, dp.Standardize())
end
if opt.zca then
    table.insert(input_preprocess, dp.ZCA())
end
if opt.lecunlcn then
    table.insert(input_preprocess, dp.GCN())
    table.insert(input_preprocess, dp.LeCunLCN{progress=true})
end

--[[data]]--
local ds = buildDataSet(opt.dataPath, opt.validRatio, opt.dataSize)

function dropout(depth)
    return opt.dropout and (opt.dropoutProb[depth] or 0) > 0 and nn.Dropout(opt.dropoutProb[depth])
end

--[[Model]]--
net = nn.Sequential()

-- convolutional and pooling layers
depth = 1
inputSize = ds:imageSize('c') --or opt.loadSize[1]
for i=1,#opt.channelSize do
    if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
        -- dropout can be useful for regularization
        net:add(nn.SpatialDropout(opt.dropoutProb[depth]))
    end
    net:add(nn.SpatialConvolution(
        inputSize, opt.channelSize[i],
        opt.kernelSize[i], opt.kernelSize[i],
        opt.kernelStride[i], opt.kernelStride[i],
        opt.padding and math.floor(opt.kernelSize[i]/2) or 0
    ))
    if opt.batchNorm then
        -- batch normalization can be awesome
        net:add(nn.SpatialBatchNormalization(opt.channelSize[i]))
    end
    net:add(nn[opt.activation]())
    if opt.poolSize[i] and opt.poolSize[i] > 0 then
        net:add(nn.SpatialMaxPooling(
            opt.poolSize[i], opt.poolSize[i],
            opt.poolStride[i] or opt.poolSize[i],
            opt.poolStride[i] or opt.poolSize[i]
        ))
    end
    inputSize = opt.channelSize[i]
    depth = depth + 1
end
-- get output size of convolutional layers
outsize = net:outside{1,ds:imageSize('c'),ds:imageSize('h'),ds:imageSize('w')}
inputSize = outsize[2]*outsize[3]*outsize[4]
dp.vprint(not opt.silent, "input to dense layers has: "..inputSize.." neurons")

net:insert(nn.Convert(ds:ioShapes(), 'bchw'), 1)

-- dense hidden layers
net:add(nn.Collapse(3))
for i,hiddenSize in ipairs(opt.hiddenSize) do
    if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
        net:add(nn.Dropout(opt.dropoutProb[depth]))
    end
    net:add(nn.Linear(inputSize, hiddenSize))
    if opt.batchNorm then
        net:add(nn.BatchNormalization(hiddenSize))
    end
    net:add(nn[opt.activation]())
    inputSize = hiddenSize
    depth = depth + 1
end

-- output layer
if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
    net:add(nn.Dropout(opt.dropoutProb[depth]))
end
net:add(nn.Linear(inputSize, #(ds:classes())))
net:add(nn.LogSoftMax())

-- RyNet
cnn = RyNet(ds)

--[[Propagators]]--
if opt.lrDecay == 'adaptive' then
    ad = dp.AdaptiveDecay{max_wait = opt.maxWait, decay_factor=opt.decayFactor}
elseif opt.lrDecay == 'linear' then
    opt.decayFactor = (opt.minLR - opt.learningRate)/opt.saturateEpoch
end

train = dp.Optimizer{
    acc_update = opt.accUpdate,
    loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
    epoch_callback = function(model, report) -- called every epoch
    if report.epoch > 0 then
        if opt.lrDecay == 'adaptive' then
            opt.learningRate = opt.learningRate*ad.decay
            ad.decay = 1
        elseif opt.lrDecay == 'schedule' and opt.schedule[report.epoch] then
            opt.learningRate = opt.schedule[report.epoch]
        elseif opt.lrDecay == 'linear' then
            opt.learningRate = opt.learningRate + opt.decayFactor
        end
        opt.learningRate = math.max(opt.minLR, opt.learningRate)
        if not opt.silent then
            print('learningRate', opt.learningRate)
        end
    end
    end,
    callback = function(model, report) -- called every batch
    -- the ordering here is important
    if opt.accUpdate then
        model:accUpdateGradParameters(model.dpnn_input, model.output, opt.learningRate)
    else
        model:updateGradParameters(opt.momentum) -- affects gradParams
        model:updateParameters(opt.learningRate) -- affects params
    end
    model:maxParamNorm(opt.maxOutNorm) -- affects params
    model:zeroGradParameters() -- affects gradParams
    end,
    feedback = dp.Confusion(),
    sampler = dp.ShuffleSampler{batch_size = opt.batchSize},
    progress = opt.progress
}
valid = ds:validSet() and dp.Evaluator{
    feedback = dp.Confusion(),
    sampler = dp.Sampler{batch_size = opt.batchSize}
}
test = ds:testSet() and dp.Evaluator{
    feedback = dp.Confusion(),
    sampler = dp.Sampler{batch_size = opt.batchSize}
}

--[[Experiment]]--
xp = dp.Experiment{
    model = net,
    optimizer = train,
    validator = ds:validSet() and valid,
    tester = ds:testSet() and test,
    observer = {
        dp.FileLogger(opt.resultsPath),
        dp.EarlyStopper{
            error_report = {'validator','feedback','confusion','accuracy'},
            maximize = true,
            max_epochs = opt.maxTries
        },
        ad
    },
    --random_seed = os.time(),
    random_seed = 2015,
    max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.useDevice)
    xp:cuda()
end

if not opt.silent then
    print"Model:"
    print(cnn)
end
xp:verbose(not opt.silent)

local start_time = os.time()
xp:run(ds)
print('Elapsed time: ' .. os.difftime(os.time(), start_time) .. ' seconds')

torch.save(paths.concat(opt.resultsPath, xp:report().id .. '_net.dat'), net)