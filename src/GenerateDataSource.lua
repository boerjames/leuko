require 'dp'
require 'torchx'
require 'image'
require './DataLoader.lua'

--[[how to use]]-- $> th main.lua [flag] [parameter]
--[[command line arguments]]
local cmd = torch.CmdLine()
cmd:text()
cmd:option('--host',                'facetag-db',       'the host connect to')
cmd:option('--dbname',              'facetag',          'the db to use')
cmd:option('--user',                'facetag',          'the user to use')
cmd:option('--password',            '',                 'the password for the user, do not fill this in, use cmd line')
cmd:option('--savePath',            '/root/deep/test/', 'the where to save artifacts')
cmd:option('--validPercentage',     0.15,               'percentage of data to use for validation')
cmd:option('--testPercentage',      0.15,               'perctage of date to use for testing')
cmd:option('--dataSize',            '{3,40,40}',        'the shape of the input data')
cmd:option('--lcn',                 false,              'apply Yann LeCun Local Contrast Normalization')
cmd:option('--silent',              false,              'dont print anything to stdout')
cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
    table.print(opt)
end
opt.dataSize = table.fromString(opt.dataSize)
collectgarbage()

--[[setup database connection]]
local env = require('luasql.postgres'):postgres()
local conn = env:connect("host=" .. opt.host .. " user=" .. opt.user .. " dbname=" .. opt.dbname .. " password=" .. opt.password)

--[[generate the datasource]]--
-- loop through the images
local image_cursor = conn:execute("select image.id, image.status from image")
local num_image_rows = image_cursor:numrows()
local crops = {}

for i = 1, num_image_rows do
    local image_res = {}
    image_cursor:fetch(image_res, "a")
    image_res["id"] = tonumber(image_res["id"])

    local eye_cursor = conn:execute("select eye_tag.id, eye_tag.image_id, eye_tag.label, eye_tag.left, eye_tag.top, eye_tag.width, eye_tag.height from eye_tag where eye_tag.image_id=" .. image_res["id"])
    local num_eye_tags = eye_cursor:numrows()
    local eye_tags = {}

    for j = 1, num_eye_tags do
        local eye_tags_tmp = {}
        eye_cursor:fetch(eye_tags_tmp, "a")
        eye_tags_tmp["id"] = tonumber(eye_tags_tmp["id"])
        eye_tags_tmp["image_id"] = tonumber(eye_tags_tmp["image_id"])
        eye_tags_tmp["left"] = tonumber(eye_tags_tmp["left"])
        eye_tags_tmp["top"] = tonumber(eye_tags_tmp["top"])
        eye_tags_tmp["width"] = tonumber(eye_tags_tmp["width"])
        eye_tags_tmp["height"] = tonumber(eye_tags_tmp["height"])

        if eye_tags_tmp["label"] == "H" or eye_tags_tmp["label"] == "L" then
            if eye_tags_tmp["width"] > 0 and eye_tags_tmp["height"] > 0 then
                table.insert(eye_tags, eye_tags_tmp)
            end
        end
    end

    print('EXAMINING THE', #eye_tags, 'EYE TAGS FROM IMAGE', image_res["id"])
    if #eye_tags == 0 then
        -- do nothing
    elseif #eye_tags == 1 then

        -- add this eye_tag to the set of crops
        table.insert(crops, eye_tags[1])
    elseif #eye_tags >= 2 then

        -- there are multiple eye_tag and need to be checked for similarity
        local tags_to_use = {}
        for e1 = 1, #eye_tags do
            for e2 = e1 + 1, #eye_tags do
                local et1, et2 = eye_tags[e1], eye_tags[e2]
                local sim, e = setsimilarity(pixelset({et1["left"], et1["top"], et1["width"], et1["height"]}, 100000, 100000), pixelset({et2["left"], et2["top"], et2["width"], et2["height"]}, 100000, 100000))
                if sim < 0.5 then
                    tags_to_use[e1] = true
                    tags_to_use[e2] = true
                elseif e == 1 then
                    tags_to_use[e1] = true
                elseif e == 2 then
                    tags_to_use[e2] = true
                end
            end
        end

        -- add the distinct eye_tags to crops
        for k, v in pairs(eye_tags) do
            if tags_to_use[k] then
                table.insert(crops, v)
            end
        end
    end
    collectgarbage()
end

local error_string = ''
for k, v in pairs(crops) do
    image_cursor = conn:execute("select image.id, image.data from image where image.id=" .. v["image_id"])
    local res = {}
    image_cursor:fetch(res, "a")

    local status, img = pcall(function() return DataLoader.imagefrompostgres(res, "data") end)
    print(status)
    if status then
        img = DataLoader.todepth(img, 3)

        if v["left"] + v["width"] <= img:size(3) and v["top"] + v["height"] <= img:size(2) then
            img = image.crop(img, v["left"], v["top"], v["left"] + v["width"], v["top"] + v["height"])
            img = image.scale(img, 40, 40)
            if v["label"] == "H" then
                image.save(opt.savePath .. "normal/" .. v["image_id"] .. "-" .. v["id"] .. ".jpg", img)
            elseif v["label"] == "L" then
                image.save(opt.savePath .. "leuko/" .. v["image_id"] .. "-" .. v["id"] .. ".jpg", img)
            end
        end
    else
        error_string = error_string .. 'image.id ' .. v["image_id"] .. ' is not a jpg' .. '\n'
    end
    print('done processing a crop')
end

torch.save('error.log', error_string, 'ascii')
