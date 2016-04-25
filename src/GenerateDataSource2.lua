require 'dp'
require 'torchx'
require 'image'
require './DataLoader.lua'
require './Similarity.lua'

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

local error_string = ''

--[[setup database connection]]
local env = require('luasql.postgres'):postgres()
local conn = env:connect("host=" .. opt.host .. " user=" .. opt.user .. " dbname=" .. opt.dbname .. " password=" .. opt.password)

--[[generate the datasource]]--
-- loop through the images
local image_cursor = conn:execute("select image.id from image")
local num_image_rows = image_cursor:numrows()
--num_image_rows = 10

for i = 1,  num_image_rows do
    local image_res = {}
    local eye_tags = {}

    image_cursor:fetch(image_res, "a")
    image_res["id"] = tonumber(image_res["id"])

    local eye_cursor = conn:execute("select eye_tag.id, eye_tag.image_id, eye_tag.left, eye_tag.top, eye_tag.width, eye_tag.height, eye_tag.label from eye_tag where eye_tag.image_id=" .. image_res["id"])
    local num_eye_tags = eye_cursor:numrows()

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
    if #eye_tags > 0 then

        local image_data_cursor = conn:execute("select image.data from image where image.id=" .. image_res["id"])
        local image_data_res = {}
        image_data_cursor:fetch(image_data_res, "a")
        local status, img = pcall(function() return DataLoader.imagefromstring(image_data_res["data"]) end)

        if status then
            local good_crops = {}

            img = DataLoader.todepth(img, 3)
            local image_width = img:size(3)
            local image_height = img:size(2)

            if #eye_tags == 1 then

                -- add this eye_tag to the set of crops
                table.insert(good_crops, eye_tags[1])

            elseif #eye_tags > 1 then

                -- there are multiple eye_tag and need to be checked for similarity
                local tags_to_use = {}
                for e1 = 1, #eye_tags do
                    for e2 = e1 + 1, #eye_tags do
                        local et1, et2 = eye_tags[e1], eye_tags[e2]
                        local sim, e = Similarity.cropsimilarity(et1, et2, image_width, image_height)
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
                    if tags_to_use[k] and v["left"] + v["width"] <= image_width and v["top"] + v["height"] <= image_height then
                        table.insert(good_crops, v)
                    end
                end

            end -- if #eye_tags == 1 elseif #eye_tags > 1

            for k, v in pairs(good_crops) do
                local crop = {v["left"], v["top"], v["width"], v["height"]}
                local outer_crop = DataLoader.outercrop(img, crop, 0.2) 
                for augmentation = 1, 10 do

                    local aug = DataLoader.augment(outer_crop, 40)
                    if v["label"] == "H" then
                        image.save(opt.savePath .. "normal/" .. v["image_id"] .. "-" .. v["id"] .. "-" .. augmentation .. ".jpg", aug)
                    elseif v["label"] == "L" then
                        image.save(opt.savePath .. "leuko/" .. v["image_id"] .. "-" .. v["id"] .. "-" .. augmentation .. ".jpg", aug)
                    end
                end
            end
        else
            error_string = error_string .. 'image ' .. image_res["id"] .. ' could not be decoded as a jpg or png\n'
        end -- if status
    end -- if #eye_tags > 0
    collectgarbage()
end -- loop over images

torch.save('error.log', error_string, 'ascii')
