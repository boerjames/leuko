require 'dp'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('$> th ModelBuilder.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')

--[[training parameters]]
cmd:option('--learningRate',        0.1,        'learning rate at t=0')
cmd:option('--lrDecay',             'linear',   'type of learning rate decay: adaptive | linear | schedule | none')
cmd:option('--minLR',               0.00001,    'minimum learning rate')
cmd:option('--saturateEpoch',       300,        'epoch at which linear decayed LR will reach minLR')
cmd:option('--schedule',            '{}',       'learning rate schedule')
cmd:option('--maxWait',             4,          'maximum number of epochs to wait for a new minima to be found after which the learning rate is decayed by decayFactor')
cmd:option('--decayFactor',         0.01,       'factor by which learning rate is decayed for adaptive decay')
cmd:option('--maxOutNorm',          1,          'max norm each layers output neuron weights')
cmd:option('--momentum',            0.1,        'momentum')
cmd:option('--batchSize',           256,        'number of examples per batch')
cmd:option('--cuda',                false,       'use CUDA')
cmd:option('--useDevice',           0,          'sets the gpu to use, use the command line to set this')
cmd:option('--maxEpoch',            200,        'maximum number of epochs to run')
cmd:option('--maxTries',            30,         'maximum number of epochs to try to find a better local minima for early-stopping')

--[[network paramters]]
cmd:option('--network',             'Custom',                       'network to use: Custom | RyaNet')
cmd:option('--channelSize',         '{8,12,16}',                    'number of output channels for each convolution layer')
cmd:option('--convStacks',          1,                              'number of convolutions before pooling on each layer')
cmd:option('--convLocal',           false,                           'first layer be a local convolution (without weight sharing)?')
cmd:option('--kernelSize',          '{3,3,3,3}',                    'kernel size of each convolution layer (h = w)')
cmd:option('--kernelStride',        '{1,1,1,1}',                    'kernel stride of each convolution layer (h = w)')
cmd:option('--padding',             true,                           'add math.floor(kernelSize/2) padding to the input of each convolution')
cmd:option('--poolSize',            '{3,3,3,3}',                    'size of the pooling of each convolution layer (h = w)')
cmd:option('--poolStride',          '{2,2,2,2}',                    'stride of the pooling of each convolution layer (h = w)')
cmd:option('--pooling',             'SpatialConvolution',            'type of pooling to use: SpatialMaxPooling | SpatialAveragePooling | SpatialConvolution')
cmd:option('--activation',          'ELU',                         'transfer function like ReLU, PReLU, RReLU, ELU, Tanh, Sigmoid')
cmd:option('--hiddenSize',          '{100}',                        'size of the dense hidden layers after the convolution')
cmd:option('--dropout',             true,                           'use dropout')
cmd:option('--dropoutProb',         '{0.1,0.5}',                    'dropout probabilities for 1) conv 2) fc')

--[[data parameters]]
cmd:option('--dataset',             'leuko-equal.t7',   'which dataset to use')
cmd:option('--standardize',         false,              'apply Standardize preprocessing')
cmd:option('--zca',                 false,              'apply Zero-Component Analysis whitening')
cmd:option('--lecunlcn',            false,              'apply Yann LeCun Local Contrast Normalization')

cmd:option('--accUpdate',           false,              'accumulate gradients inplace')
cmd:option('--progress',            true,               'print progress bar')
cmd:option('--silent',              false,              'dont print anything to stdout')
cmd:text()
opt = cmd:parse(arg or {})
if not opt.silent then
   table.print(opt)
end

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
local ds = torch.load(opt.dataset)

--[[Model]]--
local cnn

if opt.network == 'Custom' then
    cnn = nn.Sequential()
    cnn:add(nn.Convert(ds:ioShapes(), 'bchw'))

    -- convolution and pooling layers
    inputSize = ds:imageSize('c')
    local start = 1

    -- if desired, start with a locally connected convolution layer
    if opt.convLocal then
        start = start + 1
        if opt.dropout and (opt.dropoutProb[1] or 0) > 0 then
           -- dropout can be useful for regularization
           cnn:add(nn.SpatialDropout(opt.dropoutProb[1]))
        end
        cnn:add(nn.SpatialConvolutionLocal(
           inputSize, opt.channelSize[1],
           ds:imageSize('w'), ds:imageSize('h'),
           opt.kernelSize[1], opt.kernelSize[1],
           opt.kernelStride[1], opt.kernelStride[1],
           opt.padding and math.floor(opt.kernelSize[1]/2) or 0
        ))
        inputSize = opt.channelSize[1]
        cnn:add(nn[opt.activation]())
        if opt.poolSize[1] and opt.poolSize[1] > 0 then
           if opt.pooling ~= 'SpatialConvolution' then
               cnn:add(nn[opt.pooling](
               opt.poolSize[1], opt.poolSize[1],
               opt.poolStride[1] or opt.poolSize[1],
               opt.poolStride[1] or opt.poolSize[1]
           ))
           else
               cnn:add(nn.SpatialConvolution(
               opt.channelSize[1], opt.channelSize[1],
               opt.poolSize[1], opt.poolSize[1],
               opt.poolStride[1], opt.poolStride[1]
               ))
               cnn:add(nn[opt.activation]())
           end
        end
    end

    -- normal convolution and pooling layers
    for i=start,#opt.channelSize do
       for j=1,opt.convStacks do
           if opt.dropout and (opt.dropoutProb[1] or 0) > 0 then
              -- dropout can be useful for regularization
              cnn:add(nn.SpatialDropout(opt.dropoutProb[1]))
           end
           cnn:add(nn.SpatialConvolution(
              inputSize, opt.channelSize[i],
              opt.kernelSize[i], opt.kernelSize[i],
              opt.kernelStride[i], opt.kernelStride[i],
              opt.padding and math.floor(opt.kernelSize[i]/2) or 0
           ))
           inputSize = opt.channelSize[i]
           cnn:add(nn[opt.activation]())
       end
       if opt.poolSize[i] and opt.poolSize[i] > 0 then
          if opt.pooling ~= 'SpatialConvolution' then
              cnn:add(nn[opt.pooling](
              opt.poolSize[i], opt.poolSize[i],
              opt.poolStride[i] or opt.poolSize[i],
              opt.poolStride[i] or opt.poolSize[i]
          ))
          else
              cnn:add(nn.SpatialConvolution(
              opt.channelSize[i], opt.channelSize[i],
              opt.poolSize[i], opt.poolSize[i],
              opt.poolStride[i], opt.poolStride[i]
              ))
              cnn:add(nn[opt.activation]())
          end
       end
    end

    -- get output size of convolutional layers
    outsize = cnn:outside{1,ds:imageSize('c'),ds:imageSize('h'),ds:imageSize('w')}
    inputSize = outsize[2]*outsize[3]*outsize[4]

    -- dense hidden layers
    --cnn:add(nn.Collapse(3))
    cnn:add(nn.View(inputSize))
    for i,hiddenSize in ipairs(opt.hiddenSize) do
       if opt.dropout and (opt.dropoutProb[2] or 0) > 0 then
          cnn:add(nn.Dropout(opt.dropoutProb[2]))
       end
       cnn:add(nn.Linear(inputSize, hiddenSize))
       cnn:add(nn[opt.activation]())
       inputSize = hiddenSize
    end

    -- output layer
    if opt.dropout and (opt.dropoutProb[2] or 0) > 0 then
       cnn:add(nn.Dropout(opt.dropoutProb[2]))
    end
    cnn:add(nn.Linear(inputSize, #(ds:classes())))
    cnn:add(nn.LogSoftMax())
elseif not (opt.network == 'Custom') then
    print('Using network ' .. opt.network)
    cnn = require('./Models.lua')(opt.network, ds)
end

-- initialize the weights using smart initialization
cnn = require('./WeightInitialization.lua')(cnn, 'kaiming')

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
            print("learningRate", opt.learningRate)
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
   model = cnn,
   optimizer = train,
   validator = ds:validSet() and valid,
   tester = ds:testSet() and test,
   observer = {
      dp.FileLogger(),
      dp.EarlyStopper{
         error_report = {'validator','feedback','confusion','accuracy'},
         maximize = true,
         max_epochs = opt.maxTries
      },
      ad
   },
   random_seed = os.time(),
   max_epoch = opt.maxEpoch
}

--[[GPU or CPU]]--
if opt.cuda then
   require 'cutorch'
   require 'cunn'
   require 'cudnn'
   cudnn.benchmark = true
   cudnn.fastest = true
   cutorch.setDevice(opt.useDevice)
   xp:cuda()
end

if not opt.silent then
   print"Model:"
   print(cnn)
end
xp:verbose(not opt.silent)

xp:run(ds)
--[[torch.save('xp.t7',xp)

require 'cutorch'
require 'cudnn'
model = xp:model()
cudnn.convert(model, nn)
model:float()
model:clearState()
model:evaluate()
smodel = nn.Serial(model, 'torch.FloatTensor')
smodel:lightSerial()
torch.save('smodel.t7',smodel)
]]--
