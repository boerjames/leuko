
require 'dp'
require 'optim'
require 'image'
require 'torchx'

-- crop the given image on the given crop boundary, with some random given variation in translation
function randomTranslation(img, crop_boundary, variation)
    --[[
    img: the torch image to crop

    crop_boundary: coincides with crop_boundary defintion from database
     [1] left boundary
     [2] right boundary
     [3] width of crop
     [4] height of crop

    varation: the number of pixels to randomly vary
    --]]

    local x = crop_boundary[1] + (torch.uniform(0, variation) - variation / 2)
    local y = crop_boundary[2] + (torch.uniform(0, variation) - variation / 2)
    local width = crop_boundary[3]
    local height = crop_boundary[4]

    return image.crop(img, x, y, x + width, y + height)
end

-- crop the given image on the given crop boundary, with some random given variation in scale
function randomScale(img, crop_boundary, variation)
    local v = torch.uniform(0, variation) - variation / 2
    local x = crop_boundary[1] + v
    local y = crop_boundary[2] + v
    local width = crop_boundary[3] - 2 * v
    local height = crop_boundary[4] - 2 * v

    return image.crop(img, x, y, x + width, y + height)
end

function randomRotation()
end

function generateDataSet(dataPath, transformPath, dataSize)

end

function buildDataSet(dataPath, validRatio, dataSize)
    print('Loading images...')
    local c = dataSize[1]
    local h = dataSize[2]
    local w = dataSize[3]

    -- 1. Load images into input and target Tensors
    local normal = paths.indexdir(paths.concat(dataPath, 'normal')) -- 1
    local leuko = paths.indexdir(paths.concat(dataPath, 'leuko'))   -- 2

    --local size = normal:size() + leuko:size()
    local numNormal = 100--normal:size()
    local numLeuko = 100--leuko:size()
    local size = numNormal + numLeuko

    local shuffle = torch.randperm(size)
    local input = torch.FloatTensor(size, c, h, w)
    local target = torch.IntTensor(size)

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

        local idx = shuffle[i + numNormal]
        input[idx]:copy(img)
        target[idx] = 2
        collectgarbage()
    end

    -- 2. Divide into train and valid set and wrap into views
    local nValid = math.floor(size * validRatio)
    local nTrain = size - nValid

    local trainInput = dp.ImageView('bchw', input:narrow(1, 1, nTrain))
    local trainTarget = dp.ClassView('b', target:narrow(1, 1, nTrain))
    local validInput = dp.ImageView('bchw', input:narrow(1, nTrain + 1, nValid))
    local validTarget = dp.ClassView('b', target:narrow(1, nTrain + 1, nValid))

    trainTarget:setClasses({'normal', 'leuko'})
    validTarget:setClasses({'normal', 'leuko'})

    -- 3. Wrap views into datasets
    local train = dp.DataSet{inputs = trainInput, targets = trainTarget, which_set = 'train' }
    local valid = dp.DataSet{inputs = validInput, targets = validTarget, which_set = 'valid' }

    -- 4. Wrap datasets into datasource
    local ds = dp.DataSource{train_set = train, valid_set = valid }
    ds:classes{'normal', 'leuko'}
    collectgarbage()
    return ds
end

function buildDataSetPseudo(dataPath, validRatio, dataSize)
    print('Loading images...')
    local c = dataSize[1]
    local h = dataSize[2]
    local w = dataSize[3]
    validRatio = validRatio or 0.15

    -- 1. Load images into input and target Tensors
    local normal = paths.indexdir(paths.concat(dataPath, 'normal')) -- 1
    local pseudo = paths.indexdir(paths.concat(dataPath, 'pseudo')) -- 2
    local leuko = paths.indexdir(paths.concat(dataPath, 'leuko'))   -- 3

    --local size = normal:size() + pseudo:size() + leuko:size()
    local numNormal = 10--normal:size()
    local numPseudo = 10--pseudo:size()
    local numLeuko = 10--leuko:size()
    local size = numNormal + numPseudo + numLeuko

    local shuffle = torch.randperm(size)
    local input = torch.FloatTensor(size, c, h, w)
    local target = torch.IntTensor(size)

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

function determineClass(inString)
    if string.find(inString, '_uncertain_leukocoric_eye_') then return nil end
    if string.find(inString, '_leukocoric_eye_') then return 2 end
    if string.find(inString, '_iphone_white_eyes_') then return 2 end
    if string.find(inString, '_eye_') then return 1 end
    if string.find(inString, '_iphone_normal_with_flash_') then return 1 end
    if string.find(inString, '_iphone_normal_no_flash_') then return 1 end
    return nil
end


function determineClassPseudo(inString)
    if string.find(inString, '_uncertain_leukocoric_eye_') then return nil end
    if string.find(inString, '_leukocoric_eye_') then return 3 end
    if string.find(inString, '_iphone_white_eyes_') then return 2 end
    if string.find(inString, '_eye_') then return 1 end
    if string.find(inString, '_iphone_normal_with_flash_') then return 1 end
    if string.find(inString, '_iphone_normal_no_flash_') then return 1 end
    return nil
end
