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

function buildDataSet(dataPath, validRatio, dataSize)
    local c = dataSize[1]
    local h = dataSize[2]
    local w = dataSize[3]
    validRatio = validRatio or 0.15

    -- 1. Load images into input and target Tensors
    local normal = paths.indexdir(paths.concat(dataPath, 'normal')) -- 1
    local pseudo = paths.indexdir(paths.concat(dataPath, 'pseudo')) -- 2
    local leuko = paths.indexdir(paths.concat(dataPath, 'leuko'))   -- 3

    --local size = normal:size() + pseudo:size() + leuko:size()
    local numNormal = normal:size()
    local numPseudo = pseudo:size()
    local numLeuko = leuko:size()
    local size = numNormal + numPseudo + numLeuko

    local shuffle = torch.randperm(size)
    local input = torch.FloatTensor(size, c, h, w)
    local target = torch.IntTensor(size)

    print('Loading images...')
    --for i = 1, normal:size() do
    for i = 1, numNormal do
        local img = image.load(normal:filename(i))
        img = image.scale(img, h, w)

        if img:size(1) == 1 then
            local rgb = torch.Tensor(c, h, w)
            for i = 1, c do
                rgb[i] = img
            end
            img = rgb
        end

        local idx = shuffle[i]
        input[idx]:copy(img)
        target[idx] = 1
        collectgarbage()
    end

    --for i = 1, pseudo:size() do
    for i = 1, numPseudo do
        local img = image.load(pseudo:filename(i))
        img = image.scale(img, h, w)

        if img:size(1) == 1 then
            local rgb = torch.Tensor(c, h, w)
            for i = 1, c do
                rgb[i] = img
            end
            img = rgb
        end

        local idx = shuffle[i + numNormal]
        input[idx]:copy(img)
        target[idx] = 2
        collectgarbage()
    end

    --for i = 1, leuko:size() do
    for i = 1, numLeuko do
        local img = image.load(leuko:filename(i))
        img = image.scale(img, h, w)

        if img:size(1) == 1 then
            local rgb = torch.Tensor(c, h, w)
            for i = 1, c do
                rgb[i] = img
            end
            img = rgb
        end

        local idx = shuffle[i + numNormal + numPseudo]
        input[idx]:copy(img)
        target[idx] = 3
        collectgarbage()
    end

    -- 2. Divide into train and valid set and wrap into views
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
            elseif class == 1 then
                osCommand('mv ' .. filePath .. ' ' .. normalPath)
            elseif class == 2 then
                osCommand('mv ' .. filePath .. ' ' .. pseduoPath)
            elseif class == 3 then
                osCommand('mv ' .. filePath .. ' ' .. leukoPath)
            end
        end
    end
    collectgarbage()
end

function determineClass(inString)
    if string.find(inString, '_uncertain_leukocoric_eye_') then return nil end
    if string.find(inString, '_leukocoric_eye_') then return 3 end
    if string.find(inString, '_iphone_white_eyes_') then return 2 end
    if string.find(inString, '_eye_') then return 1 end
    if string.find(inString, '_iphone_normal_with_flash_') then return 1 end
    if string.find(inString, '_iphone_normal_no_flash_') then return 1 end
    return nil
end
