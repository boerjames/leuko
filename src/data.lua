--
-- Created by IntelliJ IDEA.
-- User: boer
-- Date: 10/9/15
-- Time: 10:58 AM
-- To change this template use File | Settings | File Templates.
--

require 'dp'
require 'image'
require 'torchx'
require 'util.lua'

function buildDataSet(dataPath, validRatio)
    validRatio = validRatio or 0.15

    -- 1. Load images into input and target Tensors
    local normal = paths.indexdir(paths.concat(dataPath, 'normal')) -- 1
    local pseudo = paths.indexdir(paths.concat(dataPath, 'pseudo')) -- 2
    local leuko = paths.indexdir(paths.concat(dataPath, 'leuko'))   -- 3

    local size = normal:size() + pseudo:size() + leuko:size()
    local shuffle = torch.randperm(size)
    local input = torch.FloatTensor(size, 3, 32, 32)
    local target = torch.IntTensor(size)

    for i = 1, normal:size() do
        print('Loading ' .. normal:filename(i))
        local img = image.load(normal:filename(i)):resize(3, 32, 32)
        local idx = shuffle[i]
        input[idx]:copy(img)
        target[idx] = 1
        collectgarbage()
    end

    for i = 1, pseudo:size() do
        print('Loading ' .. pseudo:filename(i))
        local img = image.load(pseudo:filename(i)):resize(3, 32, 32)
        local idx = shuffle[i + normal:size()]
        input[idx]:copy(img)
        target[idx] = 2
        collectgarbage()
    end

    for i = 1, leuko:size() do
        print('Loading ' .. leuko:filename(i))
        local img = image.load(leuko:filename(i)):resize(3, 32, 32)
        local idx = shuffle[i + normal:size() + pseudo:size()]
        input[idx]:copy(img)
        target[idx] = 3
        collectgarbage()
    end

    -- 2. Divide into trian and valid set and wrap into views
    local nValid = math.floor(size * validRatio)
    local nTrain = size - nValid

    local trainInput = dp.ImageView('bchw', input:narrow(1, 1, nTrain))
    local trainTarget = dp.ClassView('b', target:narrow(1, 1, nTrain))
    local validInput = dp.ImageView('bchw', input:narrow(1, nTrain + 1, nValid))
    local validTarget = dp.ClassView('b', target:narrow(1, nTrain + 1, nValid))

    trainTarget:setClasses({'normal', 'pseudo', 'leuko'})
    validTarget:setClasses({'normal', 'pseudo', 'leuko'})

    -- 3. Wrap views into datasets
    local train = dp.DataSet{inputs = trainInput, targets = trainTarget, which_set = 'train' }
    local valid = dp.DataSet{inputs = validInput, targets = validTarget, which_set = 'valid' }

    -- 4. Wrap datasets into datasource
    local ds = dp.DataSource{train_set = train, valid_set = valid }
    ds:classes{'normal', 'pseudo', 'leuko' }
    collectgarbage()
    return ds
end

function moveData(folder)
    local normalPath = folder .. '/normal'
    local pseduoPath = folder .. '/pseudo'
    local leukoPath = folder .. '/leuko'

    osCommand('mkdir ' .. normalPath)
    osCommand('mkdir ' .. pseduoPath)
    osCommand('mkdir ' .. leukoPath)

    for file in io.popen('ls ' .. folder):lines() do
        if string.find(file, '%.jpg$') or string.find(file, '%.png$') then
            local class = determineClass(file)
            local filePath = path.join(folder, file)
            print('Moving ' .. filePath)
            if class == nil then
                osCommand('rm ' .. filePath)
            elseif class == 0 then
                osCommand('mv ' .. filePath .. ' ' .. normalPath)
            elseif class == 1 then
                osCommand('mv ' .. filePath .. ' ' .. pseduoPath)
            elseif class == 2 then
                osCommand('mv ' .. filePath .. ' ' .. leukoPath)
            end
        end
    end
    collectgarbage()
end

function determineClass(inString)
    if string.find(inString, '_uncertain_leukocoric_eye_') then return nil end
    if string.find(inString, '_leukocoric_eye_') then return 2 end
    if string.find(inString, '_iphone_white_eyes_') then return 1 end
    if string.find(inString, '_eye_') then return 0 end
    if string.find(inString, '_iphone_normal_with_flash_') then return 0 end
    if string.find(inString, '_iphone_normal_no_flash_') then return 0 end
    return nil
end