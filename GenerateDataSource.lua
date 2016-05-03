require 'dp'
require 'xlua'
require 'torchx'
require 'image'

local function pixelset(left, top, width, height, image_width, image_height)
    local set = {}
    for j = top, top + height - 1 do
        for i = left, left + width - 1 do
            local pixelnum = (j - 1) * image_width + i
            set[pixelnum] = true
        end
    end
    return set
end

local function setsimilarity(s1, s2)
    local u, i = {}, {}
    local s1len, s2len, ulen, ilen = 0, 0, 0, 0

    -- build union set
    for k,v in pairs(s1) do
        if v then
            u[k] = v
            s1len = s1len + 1
        end
    end

    for k,v in pairs(s2) do
        if v then
            u[k] = v
            s2len = s2len + 1
        end
    end

    -- build intersection set
    for k1,v1 in pairs(s1) do
        local v2 = s2[k1]
        if v1 and v2 then
            i[k1] = v1
        end
    end

    for k,v in pairs(u) do
        ulen = ulen + 1
    end

    for k,v in pairs(i) do
        ilen = ilen + 1
    end

    local sim = ilen / ulen
    if s1len > s2len then
        return sim, 1
    else
        return sim, 2
    end

end

local function cropsimilarity(et1, et2, image_width, image_height)
    local pixelset1 = pixelset(et1["left"], et1["top"], et1["width"], et1["height"], image_width, image_height)
    local pixelset2 = pixelset(et2["left"], et2["top"], et2["width"], et2["height"], image_width, image_height)
    return setsimilarity(pixelset1, pixelset2)
end

local function imagefromstring(rawstr)
    local str = rawstr:sub(3, rawstr:len())
    local bytes = torch.ByteTensor(torch.ByteStorage():string(str:fromhex()))
    local img = image.decompress(bytes)
    return img
end

local function outercrop(img, crop, delta)
    local x1, y1, w, h = unpack(crop)
    local x2, y2 = x1 + w, y1 + h
    local deltax, deltay = delta * w, delta * h

    x1, x2 = x1 - deltax, x2 + deltax
    y1, y2 = y1 - deltay, y2 + deltay

    x1, x2 = math.max(x1, 1), math.min(x2, img:size(3))
    y1, y2 = math.max(y1, 1), math.min(y2, img:size(2))

    return image.crop(img, x1, y1, x2, y2)
end

local function augment(img, dim)
    local r = math.random(1, dim / 3)
    local aug = image.scale(img, dim + r, dim + r)

    -- random rotation
    aug = image.rotate(aug, math.random(0, math.rad(360)))

    -- random crop and scale
    local x1, y1 = math.ceil(math.random() * r), math.ceil(math.random() * r)
    local x2, y2 = aug:size(3) - math.ceil(math.random() * r), aug:size(2) - math.ceil(math.random() * r)
    x1, y1 = math.max(1, x1), math.max(1, y1)
    x2, y2 = math.min(aug:size(3), x2), math.min(aug:size(2), y2)
    aug = image.crop(aug, x1, y1, x2, y2)

    -- random flip, probably pointless
    if math.random() > 0.5 then aug = image.vflip(aug) end
    if math.random() > 0.5 then aug = image.hflip(aug) end

    return image.scale(aug, dim, dim)
end

--[[how to use]]-- $> th main.lua [flag] [parameter]
--[[command line arguments]]
local cmd = torch.CmdLine()
cmd:text()
cmd:option('--savePath',            '/root/shared/data',   'the where to save artifacts')
cmd:option('--host',                'facetag-db',           'the host connect to')
cmd:option('--dbname',              'facetag',              'the db to use')
cmd:option('--user',                'facetag',              'the user to use')
cmd:option('--password',            '',                     'the password for the user, do not fill this in, use cmd line')
cmd:option('--validPercentage',     0.15,                   'percentage of data to use for validation')
cmd:option('--testPercentage',      0.15,                   'perctage of date to use for testing')
cmd:option('--numVariants',         10,                     'the number of variants to create for an eye tag')
cmd:option('--dataSize',            '{3,40,40}',            'the shape of the input data')
cmd:option('--lcn',                 false,                  'apply Yann LeCun Local Contrast Normalization')
cmd:option('--silent',              false,                  'dont print anything to stdout')
cmd:text()
local opt = cmd:parse(arg or {})
if not opt.silent then
    table.print(opt)
end
opt.dataSize = table.fromString(opt.dataSize)
collectgarbage()

local normalPath = paths.concat(opt.savePath, 'normal')
local leukoPath = paths.concat(opt.savePath, 'leuko')
dp.mkdir(normalPath)
dp.mkdir(leukoPath)

local error_string = ''

--[[setup database connection]]
local env = require('luasql.postgres'):postgres()
local conn = env:connect("host=" .. opt.host .. " user=" .. opt.user .. " dbname=" .. opt.dbname .. " password=" .. opt.password)

--[[generate the datasource]]--
-- loop through the images
local num_eye_tags_total, eye_tags_counter = conn:execute("select eye_tag.id from eye_tag"):numrows(), 0
local image_cursor = conn:execute("select image.id from image")
local num_image_rows = image_cursor:numrows()
--num_image_rows = 10

if not silent then print('Generating eye crops from ' .. opt.host) end
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
        if not silent then
            eye_tags_counter = eye_tags_counter + 1
            xlua.progress(eye_tags_counter, num_eye_tags_total)
        end
    end

    --print('EXAMINING THE', #eye_tags, 'EYE TAGS FROM IMAGE', image_res["id"])
    if #eye_tags > 0 then

        local image_data_cursor = conn:execute("select image.data from image where image.id=" .. image_res["id"])
        local image_data_res = {}
        image_data_cursor:fetch(image_data_res, "a")
        local status, img = pcall(function() return imagefromstring(image_data_res["data"]) end)

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
                        local sim, e = cropsimilarity(et1, et2, image_width, image_height)
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
                local outer_crop = outercrop(img, crop, 0.2)
                for variant = 1, opt.numVariants do

                    local var = augment(outer_crop, 40)
                    if v["label"] == "H" then
                        image.save(normalPath .. "/" .. v["image_id"] .. "-" .. v["id"] .. "-" .. variant .. ".jpg", var)
                    elseif v["label"] == "L" then
                        image.save(leukoPath .. "/" .. v["image_id"] .. "-" .. v["id"] .. "-" .. variant .. ".jpg", var)
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
