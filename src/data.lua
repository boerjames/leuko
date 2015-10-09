--
-- Created by IntelliJ IDEA.
-- User: boer
-- Date: 10/9/15
-- Time: 10:58 AM
-- To change this template use File | Settings | File Templates.
--

require 'image'
require 'xlua'

require 'util.lua'

-- todo: return torch.LongStorage [#images, #channels, width, height]
function buildData(folder, width, height)
    local data = {}
    local i = 0
    --local numFiles = osCommand('ls ' .. folder .. ' | wc -w')
    for file in io.popen('ls ' .. folder):lines() do
        i = i + 1
        --xlua.progress(i, numFiles)
        if string.find(file, '%.jpg$') or string.find(file, '%.png$') then
            local class = determineClass(file)
            if class ~= nil then
                local filePath = path.join(folder, file)
                data[1] = image.scale(image.load(filePath, 3), width, height)
                data[2] = class
            end
        end
    end
    return data
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

