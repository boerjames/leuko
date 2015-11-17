--
-- Created by IntelliJ IDEA.
-- User: boer
-- Date: 11/2/15
-- Time: 3:45 PM
-- To change this template use File | Settings | File Templates.
--

require 'dp'
require 'optim'
require 'extraction.lua'

function iTorchPrepare()
    local path = '/home/boer/save/rynet/log'
    local reports = loadReports(path)

    local train, valid = getLearningAccuracy(reports)
    torch.save('/home/boer/leuko/itorch/train.t7', train)
    torch.save('/home/boer/leuko/itorch/valid.t7', valid)
end

function loadReports(resultsPath)
    local reports = {}
    local logs = paths.indexdir(resultsPath, 'dat')
    for i = 1, logs:size() do
        if string.find(logs:filename(i), 'report') then
            local report = torch.load(logs:filename(i))
            table.insert(reports, report)
        end
    end

    table.sort(reports, function(a,b) return a.epoch < b.epoch end)
    return reports
end

function getLearningAccuracy(reports)
    local train = {}
    local valid = {}
    for i = 1, #reports do
        table.insert(train, reports[i].optimizer.feedback.confusion.accuracy)
        table.insert(valid, reports[i].validator.feedback.confusion.accuracy)
    end
    return train, valid
end

--[[ a report contains the following structure
optimizer       {
  batch_duration : 0.041701430851712
  sampler :
    {
      batch_size : 10
    }
  feedback :
    {
      confusion :
        {
          classes :
            {
              1 : "normal"
              2 : "pseudo"
              3 : "leuko"
            }
          avg_per_class_accuracy : 0.97788653771083
          accuracy : 0.98444790046656
          matrix : LongTensor - size: 3x3
          per_class :
            {
              accuracy : FloatTensor - size: 3
              union_accuracy : FloatTensor - size: 3
              avg :
                {
                  accuracy : 0.97788653771083
                  union_accuracy : 0.95926749706268
                }
            }
        }
      n_sample : 25720
    }
  batch_speed : 23.979992522461
  name : "optimizer"
  loss : 0.0050794887136533
  example_speed : 239.79992522461
  epoch_duration : 107.2560801506
}
validator       {
  batch_duration : 0.021368957826219
  sampler :
    {
      batch_size : 10
    }
  feedback :
    {
      confusion :
        {
          classes :
            {
              1 : "normal"
              2 : "pseudo"
              3 : "leuko"
            }
          avg_per_class_accuracy : 0.97860527038574
          accuracy : 0.98413397972675
          matrix : LongTensor - size: 3x3
          per_class :
            {
              accuracy : FloatTensor - size: 3
              union_accuracy : FloatTensor - size: 3
              avg :
                {
                  accuracy : 0.97860527038574
                  union_accuracy : 0.95969414710999
                }
            }
        }
      n_sample : 4538
    }
  name : "validator"
  batch_speed : 46.796854022193
  example_speed : 467.76238667999
  epoch_duration : 9.7015068531036
}
epoch   99
id      ubuntu:1446502179:1
random_seed     2015
description
]]--