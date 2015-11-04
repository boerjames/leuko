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
    torch.save('/home/boer/iTorch/leuko/train.t7', train)
    torch.save('/home/boer/iTorch/leuko/valid.t7', valid)
end