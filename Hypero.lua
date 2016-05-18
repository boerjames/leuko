require 'dp'
require 'hypero'

require './Extensions.lua'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('Example:')
cmd:text('$> th Hypero.lua --useDevice 1')
cmd:text('Options:')

-- hypero database parameters
cmd:option('--batteryName',             'leuko',                'name of battery of experiments to be run')
cmd:option('--versionDesc',             'testing2',              'neural network version')

-- training options
cmd:option('--maxHex',                  300,                     'maximum number of hyper-experiments to train (from this script)')
cmd:option('--cuda',                    true,                   'use CUDA')
cmd:option('--useDevice',               0,                      'sets the device (GPU) to use, please set using cmd arg')
cmd:option('--maxEpoch',		        150,                    'maximum number of epochs to run')
cmd:option('--maxTries',                15,                     'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--startLR',                 '{0.001, 1}',           'learning rate at t=0 (log-uniform {log(min), log(max)})')
cmd:option('--minLR',                   '{0.001, 1}',           'minimum LR = minLR*startLR (log-uniform {log(min), log(max)})')
cmd:option('--satEpoch',                '{150, 50}',            'epoch at which linear decayed LR will reach minLR*startLR (normal {mean, std})')
cmd:option('--maxOutNorm',              '{1, 2, 3, 4}',         'max norm each layers output neuron weights (categorical)')
cmd:option('--momentum',                '{0.0, 0.9}',           'momentum (uniform)')
cmd:option('--batchSize',               '{512, 1024}',          'number of examples per batch (categorical)')

-- convolution options
cmd:option('--convolutionStacks',               '{2, 2}',               'number of convolutions before pooling (random int)')
cmd:option('--convolutionKernelSize',           '{3}',               'possible sizes of convolution kernels (categorical)')
cmd:option('--startConvolutionFilters',         '{8, 8}',              'starting number of convolution filters (random int)')
cmd:option('--finalConvolutionFilters',         '{8, 8}',              'final number of convolution filters (random int)')
cmd:option('--numConvolutionLayers',            '{2, 4}',               'number of convolution layers (random int)')
cmd:option('--convDropoutProb',                 '{0.0, 0.0}',           'probabilities of convolution dropout layer (uniform)')

-- activation options
cmd:option('--activation',      '{"ReLU","PReLU","RReLU","ELU"}',     'activation to use (categorical)')

-- pooling options
cmd:option('--poolSize',        '{2, 3}',                                                                   'pooling size (categorical)')
cmd:option('--poolMethod',      '{"SpatialMaxPooling", "SpatialConvolution"}',     'pooling method (categorical)')

cmd:option('--numFCLayers',             '{1, 2}',               'number of fully connected layers')
cmd:option('--numFCNeurons',            '{10, 50}',           'number of neurons per fully connected layer')
cmd:option('--fcDropoutProb',           '{0.3, 0.5}',           'probabilities of fully connected dropout')

cmd:option('--progress',    true,       'display progress bar')
cmd:option('--silent',      false,      'dont print anything to stdout')
cmd:option('--dropout',       '{true, false}',   'apply dropout or not (categorical)')
cmd:text()

hopt = cmd:parse(arg or {})
hopt.convolutionStacks              = dp.returnString(hopt.convolutionStacks)
hopt.convolutionKernelSize          = dp.returnString(hopt.convolutionKernelSize)
hopt.startConvolutionFilters        = dp.returnString(hopt.startConvolutionFilters)
hopt.finalConvolutionFilters        = dp.returnString(hopt.finalConvolutionFilters)
hopt.numConvolutionLayers           = dp.returnString(hopt.numConvolutionLayers)
hopt.convDropoutProb                = dp.returnString(hopt.convDropoutProb)

hopt.activation                 = dp.returnString(hopt.activation)

hopt.poolSize                   = dp.returnString(hopt.poolSize)
hopt.poolMethod                 = dp.returnString(hopt.poolMethod)

hopt.numFCLayers                = dp.returnString(hopt.numFCLayers)
hopt.numFCNeurons               = dp.returnString(hopt.numFCNeurons)
hopt.fcDropoutProb              = dp.returnString(hopt.fcDropoutProb)

hopt.startLR                    = dp.returnString(hopt.startLR)
hopt.minLR                      = dp.returnString(hopt.minLR)
hopt.satEpoch                   = dp.returnString(hopt.satEpoch)
hopt.maxOutNorm                 = dp.returnString(hopt.maxOutNorm)
hopt.momentum                   = dp.returnString(hopt.momentum)
hopt.batchSize                  = dp.returnString(hopt.batchSize)
hopt.dropout                    = dp.returnString(hopt.dropout)


function buildExperiment(opt, ds)

    --[[Model]]--
    local model = nn.Sequential()
    model:add(nn.Convert(ds:ioShapes(), 'bchw'))

    -- convolution layers
    local inputSize = ds:imageSize('c')
    local convStepSize = math.floor((opt.finalConvolutionFilters - opt.startConvolutionFilters) / (opt.numConvolutionLayers - 1))
    local conv = {}
    for i=1,opt.numConvolutionLayers do
        if i == 1 then conv[i] = opt.startConvolutionFilters
        elseif i == opt.numConvolutionLayers then conv[i] = opt.finalConvolutionFilters
        else conv[i] = conv[i-1] + convStepSize end
    end

    for i=1,#conv do
        for j=1,opt.convolutionStacks do
            if opt.dropout then model:add(nn.SpatialDropout(opt.convDropoutProb)) end
            model:add(nn.SpatialConvolution(
               inputSize, conv[i],
               opt.convolutionKernelSize, opt.convolutionKernelSize,
               1, 1,
               math.floor(opt.convolutionKernelSize/2)
            ))
            inputSize = conv[i]
            model:add(nn[opt.activation]())
        end

       if opt.poolMethod == 'SpatialConvolution' then
           model:add(nn.SpatialConvolution(
               conv[i], conv[i],
               opt.poolSize, opt.poolSize,
               2, 2,
               math.floor(opt.poolSize/2)
           ))
           model:add(nn[opt.activation]())
       else
           model:add(nn[opt.poolMethod](
               opt.poolSize, opt.poolSize,
               2, 2
           ))

       end
    end

    outputSize = model:outside{1, ds:imageSize('c'), ds:imageSize('h'), ds:imageSize('w')}
    inputSize = outputSize[2] * outputSize[3] * outputSize[4]

    -- fully connected layers
    --model:add(nn.Collapse(3))
    model:add(nn.View(inputSize))
    for i=1,opt.numFCLayers do
        if opt.dropout then model:add(nn.Dropout(opt.fcDropoutProb)) end

        model:add(nn.Linear(inputSize, opt.numFCNeurons))

        model:add(nn[opt.activation]())
        inputSize = opt.numFCNeurons
    end

    -- output layer
    model:add(nn.Linear(inputSize, #(ds:classes())))
    model:add(nn.LogSoftMax())

    -- initialize weights
    model = require('./WeightInitialization.lua')(model, 'kaiming')


   --[[Propagators]]--

   -- linear decay
   opt.learningRate = opt.startLR
   opt.decayFactor = (opt.minLR - opt.learningRate) / opt.satEpoch
   opt.lrs = {}

   local train = dp.Optimizer{
      acc_update = opt.accUpdate,
      loss = nn.ModuleCriterion(nn.ClassNLLCriterion(), nil, nn.Convert()),
      epoch_callback = function(model, report) -- called every epoch
         -- learning rate decay
         if report.epoch > 0 then
            opt.lrs[report.epoch] = opt.learningRate
            opt.learningRate = opt.learningRate + opt.decayFactor
            opt.learningRate = math.max(opt.minLR, opt.learningRate)
            if not opt.silent then
               print('learningRate', opt.learningRate)
            end
         end
      end,
      callback = function(model, report) -- called for every batch
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
   local valid = dp.Evaluator{
      feedback = dp.Confusion(),
      sampler = dp.Sampler{batch_size = opt.batchSize}
   }
   local test = dp.Evaluator{
      feedback = dp.Confusion(),
      sampler = dp.Sampler{batch_size = opt.batchSize}
   }

   --[[Experiment]]--
   -- this will be used by hypero
   local hlog = dp.HyperLog()

   local xp = dp.Experiment{
      model = model,
      optimizer = train,
      validator = valid,
      tester = test,
      observer = {
         hlog,
         dp.EarlyStopper{
            error_report = {'validator','feedback','confusion','accuracy'},
            maximize = true,
            max_epochs = opt.maxTries
         }
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

   xp:verbose(not opt.silent)
   if not opt.silent then
      print"Model :"
      print(model)
   end

   return xp, hlog
end

--[[hypero]]--
conn = hypero.connect()
bat = conn:battery(hopt.batteryName, hopt.versionDesc)
hs = hypero.Sampler()

-- this allows the hyper-param sampler to be bypassed via cmd-line
function ntbl(param)
   return torch.type(param) ~= 'table' and param
end

-- helper that allows categorical sampling to be uniform
local function evenCategorical(t)
    local count = table.length(t)
    local tt = {}
    for i=1,count do table.insert(tt,1) end
    return tt, t
end

-- existing dataset to use
print('Loading dataset...')
local ds = torch.load('leuko-equal.t7')

-- loop over experiments
for i=1,hopt.maxHex do
   collectgarbage()
   local hex = bat:experiment()
   local opt = _.clone(hopt)

   -- hyper-parameters
   local hp = {}

   hp.convolutionStacks         = ntbl(opt.convolutionStacks)           or hs:randint(unpack(opt.convolutionStacks))
   hp.convolutionKernelSize     = ntbl(opt.convolutionKernelSize)       or hs:categorical(evenCategorical(opt.convolutionKernelSize))
   hp.numConvolutionLayers      = ntbl(opt.numConvolutionLayers)        or hs:randint(unpack(opt.numConvolutionLayers))
   hp.startConvolutionFilters   = ntbl(opt.startConvolutionFilters)     or hs:randint(unpack(opt.startConvolutionFilters))
   hp.finalConvolutionFilters   = ntbl(opt.finalConvolutionFilters)     or hs:randint(unpack(opt.finalConvolutionFilters))
   hp.convDropoutProb           = ntbl(opt.convDropoutProb)             or hs:uniform(unpack(opt.convDropoutProb))

   hp.activation                = ntbl(opt.activation)                  or hs:categorical(evenCategorical(opt.activation))

   hp.poolSize                  = ntbl(opt.poolSize)                    or hs:categorical(evenCategorical(opt.poolSize))
   hp.poolMethod                = ntbl(opt.poolMethod)                  or hs:categorical(evenCategorical(opt.poolMethod))

   hp.numFCLayers               = ntbl(opt.numFCLayers)                 or hs:randint(unpack(opt.numFCLayers))
   hp.numFCNeurons              = ntbl(opt.numFCNeurons)                or hs:randint(unpack(opt.numFCNeurons))
   hp.fcDropoutProb             = ntbl(opt.fcDropoutProb)               or hs:uniform(unpack(opt.fcDropoutProb))

   hp.startLR                   = ntbl(opt.startLR)                     or hs:logUniform(math.log(opt.startLR[1]), math.log(opt.startLR[2]))
   hp.minLR                     = (ntbl(opt.minLR)                      or hs:logUniform(math.log(opt.minLR[1]), math.log(opt.minLR[2]))) * hp.startLR
   hp.satEpoch                  = ntbl(opt.satEpoch)                    or hs:normal(unpack(opt.satEpoch))
   hp.momentum                  = ntbl(opt.momentum)                    or hs:uniform(unpack(opt.momentum))
   hp.maxOutNorm                = ntbl(opt.maxOutNorm)                  or hs:categorical(evenCategorical(opt.maxOutNorm))
   hp.batchSize                 = ntbl(opt.batchSize)                   or hs:categorical(evenCategorical(opt.batchSize))
   hp.dropout                   = ntbl(opt.dropout)                     or hs:categorical(evenCategorical(opt.dropout))

   hp.finalConvolutionFilters   = math.max(hp.startConvolutionFilters, hp.finalConvolutionFilters)

   for k,v in pairs(hp) do opt[k] = v end

   if not opt.silent then
      table.print(opt)
   end

   -- build dp experiment
   local xp, hlog = buildExperiment(opt, ds)

   -- more hyper-parameters
   hp.seed = xp:randomSeed()
   hex:setParam(hp)

   -- meta-data
   local md = {}
   md.name = xp:name()
   md.hostname = os.hostname()
   md.dataset = torch.type(ds)

   if not opt.silent then
      table.print(md)
   end

   md.modelstr = tostring(xp:model())
   hex:setMeta(md)

   -- run the experiment
   local success, err = pcall(function() xp:run(ds) end )

   -- results
   if success then
      res = {}
      res.trainCurve = hlog:getResultByEpoch('optimizer:feedback:confusion:accuracy')
      res.validCurve = hlog:getResultByEpoch('validator:feedback:confusion:accuracy')
      res.testCurve = hlog:getResultByEpoch('tester:feedback:confusion:accuracy')
      res.trainAcc = hlog:getResultAtMinima('optimizer:feedback:confusion:accuracy')
      res.validAcc = hlog:getResultAtMinima('validator:feedback:confusion:accuracy')
      res.testAcc = hlog:getResultAtMinima('tester:feedback:confusion:accuracy')
      res.lrs = opt.lrs
      res.minimaEpoch = hlog.minimaEpoch
      hex:setResult(res)

      if not opt.silent then
         table.print(res)
      end
   else
      print(err)
   end
end
