require 'dp'
require 'hypero'

--[[command line arguments]]--
cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST dataset Image Classification using MLP Training')
cmd:text('Example:')
cmd:text('$> th HyperoTest.lua --batchSize 128 --momentum 0.5')
cmd:text('Options:')

cmd:option('--batteryName', 'age',    'name of battery of experiments to be run')
cmd:option('--versionDesc', 'v6',       'neural network version')

cmd:option('--maxHex',      10,    'maximum number of hyper-experiments to train (from this script)')
cmd:option('--cuda',        true,   'use CUDA')
cmd:option('--useDevice',   0,      'sets the device (GPU) to use, please set using cmd arg')
cmd:option('--maxEpoch',    150,    'maximum number of epochs to run')
cmd:option('--maxTries',    10,     'maximum number of epochs to try to find a better local minima for early-stopping')
cmd:option('--progress',    true,   'display progress bar')
cmd:option('--silent',      false,  'dont print anything to stdout')

cmd:option('--convolutionKernelSize',       '{0, 0, 1}',        'size of convolution kernels')
cmd:option('--convolutionKernelSizeOpt',    '{3, 5, 7}',        'possible sizes of convolution kernels')
cmd:option('--numConvolutionFilters',       '{0, 0, 0, 1}',     'number of convolution filters per convolution layer')
cmd:option('--numConvolutionFiltersOpt',    '{8, 12, 16, 32}',  'possible number of convolution filters per convolution layer')
cmd:option('--numConvolutionFiltersInc',    '{1, 0}',           'number of convolution filters double each layer (no or yes)')
cmd:option('--numConvolutionLayers',        '{4, 4}',           'number of convolution layers')
cmd:option('--convolutionDropoutP',         '{0.1, 0.3}',       'probabilities of convolution dropout layer')

cmd:option('--numFCLayers',             '{1, 1}',           'number of fully connected layers')
cmd:option('--numFCNeurons',            '{126, 128}',        'number of neurons per fully connected layer')
cmd:option('--fcDropoutP',              '{0.3, 0.4}',       'probabilities of fully connected dropout')

cmd:option('--startLR',                 '{0.001, 1}',       'learning rate at t=0 (log-uniform {log(min), log(max)})')
cmd:option('--minLR',                   '{0.001, 1}',       'minimum LR = minLR*startLR (log-uniform {log(min), log(max)})')
cmd:option('--satEpoch',                '{150, 50}',        'epoch at which linear decayed LR will reach minLR*startLR (normal {mean, std})')
cmd:option('--maxOutNorm',              '{1, 1, 1}',        'max norm each layers output neuron weights (categorical)')
cmd:option('--momentum',                '{1, 1, 1}',        'momentum (categorical)')
cmd:option('--batchSize',               '{0, 0, 1}',        'number of examples per batch (categorical)')
cmd:option('--extra',                   '{0, 1}',           'apply nothing or dropout (categorical)')
cmd:text()

hopt = cmd:parse(arg or {})
hopt.convolutionKernelSize      = dp.returnString(hopt.convolutionKernelSize)
hopt.convolutionKernelSizeOpt   = dp.returnString(hopt.convolutionKernelSizeOpt)
hopt.numConvolutionFilters      = dp.returnString(hopt.numConvolutionFilters)
hopt.numConvolutionFiltersOpt   = dp.returnString(hopt.numConvolutionFiltersOpt)
hopt.numConvolutionFiltersInc   = dp.returnString(hopt.numConvolutionFiltersInc)
hopt.numConvolutionLayers       = dp.returnString(hopt.numConvolutionLayers)
hopt.convolutionDropoutP        = dp.returnString(hopt.convolutionDropoutP)

hopt.numFCLayers                = dp.returnString(hopt.numFCLayers)
hopt.numFCNeurons               = dp.returnString(hopt.numFCNeurons)
hopt.fcDropoutP                 = dp.returnString(hopt.fcDropoutP)

hopt.startLR                    = dp.returnString(hopt.startLR)
hopt.minLR                      = dp.returnString(hopt.minLR)
hopt.satEpoch                   = dp.returnString(hopt.satEpoch)
hopt.maxOutNorm                 = dp.returnString(hopt.maxOutNorm)
hopt.momentum                   = dp.returnString(hopt.momentum)
hopt.batchSize                  = dp.returnString(hopt.batchSize)
hopt.extra                      = dp.returnString(hopt.extra)

--[[ dp ]]--
function buildExperiment(opt, ds)

    --[[Model]]--
    local model = nn.Sequential()
    model:add(nn.Convert(ds:ioShapes(), 'bchw'))
    local inputSize = ds:imageSize('c')
    local outputSize = opt.numConvolutionFilters

    -- convolution layers
    for i=1,opt.numConvolutionLayers do
        if opt.extra == 'dropout' then
            model:add(nn.SpatialDropout(opt.convolutionDropoutP))
        end
        
        model:add(nn.SpatialConvolution(
            inputSize, outputSize,
            opt.convolutionKernelSize, opt.convolutionKernelSize,
            1, 1,
            math.floor(opt.convolutionKernelSize / 2)
        ))

        model:add(nn.ReLU())
        model:add(nn.SpatialMaxPooling(2, 2, 2, 2))

        inputSize = outputSize
        if opt.numConvolutionFiltersInc == 'double' then
            outputSize = 2 * outputSize
        end
    end

    outputSize = model:outside{1, ds:imageSize('c'), ds:imageSize('h'), ds:imageSize('w')}
    inputSize = outputSize[2] * outputSize[3] * outputSize[4]

    -- fully connected layers
    model:add(nn.Collapse(3))
    for i=1,opt.numFCLayers do
        if opt.extra == 'dropout' then
            model:add(nn.Dropout(opt.fcDropoutP))
        end

        model:add(nn.Linear(inputSize, opt.numFCNeurons))

        model:add(nn.ReLU())
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
   opt.decayFactor = (opt.minLR - opt.learningRate)/opt.satEpoch
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
               print("learningRate", opt.learningRate)
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

-- existing dataset to use
print('Loading dataset...')
local ds = torch.load('age-dataset128.t7')

-- loop over experiments
for i=1,hopt.maxHex do
   collectgarbage()
   local hex = bat:experiment()
   local opt = _.clone(hopt)

   -- hyper-parameters
   local hp = {}
   hp.convolutionKernelSize     = ntbl(opt.convolutionKernelSize)       or hs:categorical(opt.convolutionKernelSize, opt.convolutionKernelSizeOpt)
   hp.numConvolutionFilters     = ntbl(opt.numConvolutionFilters)       or hs:categorical(opt.numConvolutionFilters, opt.numConvolutionFiltersOpt)
   hp.numConvolutionFiltersInc  = ntbl(opt.numConvolutionFiltersInc)    or hs:categorical(opt.numConvolutionFiltersInc, {'no', 'double'})
   hp.numConvolutionLayers      = ntbl(opt.numConvolutionLayers)        or hs:randint(opt.numConvolutionLayers[1], opt.numConvolutionLayers[2])
   hp.convolutionDropoutP       = ntbl(opt.convolutionDropoutP)         or hs:uniform(opt.convolutionDropoutP[1], opt.convolutionDropoutP[2])

   hp.numFCLayers           = ntbl(opt.numFCLayers)             or hs:randint(opt.numFCLayers[1], opt.numFCLayers[2])
   hp.numFCNeurons          = ntbl(opt.numFCNeurons)            or hs:randint(opt.numFCNeurons[1], opt.numFCNeurons[2])
   hp.fcDropoutP            = ntbl(opt.fcDropoutP)              or hs:uniform(opt.fcDropoutP[1], opt.fcDropoutP[2])

   hp.startLR               = ntbl(opt.startLR)                 or hs:logUniform(math.log(opt.startLR[1]), math.log(opt.startLR[2]))
   hp.minLR                 = (ntbl(opt.minLR)                  or hs:logUniform(math.log(opt.minLR[1]), math.log(opt.minLR[2]))) * hp.startLR
   hp.satEpoch              = ntbl(opt.satEpoch)                or hs:normal(unpack(opt.satEpoch))
   hp.momentum              = ntbl(opt.momentum)                or hs:categorical(opt.momentum, {0.1, 0.5, 0.9})
   hp.maxOutNorm            = ntbl(opt.maxOutNorm)              or hs:categorical(opt.maxOutNorm, {1, 2, 4})
   hp.batchSize             = ntbl(opt.batchSize)               or hs:categorical(opt.batchSize, {128, 256, 512})
   hp.extra                 = ntbl(opt.extra)                   or hs:categorical(opt.extra, {'none', 'dropout'})

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
