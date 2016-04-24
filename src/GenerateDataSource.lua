require 'dp'
require 'torchx'
require 'image'
require './DataLoader.lua'

local function pixelset(crop, image_width, image_height)
    local left, top, width, height = unpack(crop)
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

    local overlap = ilen / ulen
    if s1len > s2len then
        return overlap, 1
    else
        return overlap, 2
    end

end


--[[how to use]]-- $> th main.lua [flag] [parameter]

--[[command line arguments]]
local cmd = torch.CmdLine()
cmd:text()
cmd:option('--host',                'facetag-db',   'the host connect to')
cmd:option('--dbname',              'facetag',      'the db to use')
cmd:option('--user',                'facetag',      'the user to use')
cmd:option('--password',            '',             'the password for the user, do not fill this in, use cmd line')
cmd:option('--validPercentage',     0.15,           'percentage of data to use for validation')
cmd:option('--testPercentage',      0.15,           'perctage of date to use for testing')
cmd:option('--dataSize',            '{3,40,40}',    'the shape of the input data')
cmd:option('--lcn',                 false,          'apply Yann LeCun Local Contrast Normalization')
cmd:option('--silent',              false,          'dont print anything to stdout')
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

for k, v in pairs(crops) do
    print(k)
    print(v)
    image_cursor = conn:execute("select image.id, image.data from image where image.id=" .. v["image_id"])
    local res = {}
    image_cursor:fetch(res, "a")
    
    local img = DataLoader.imagefrompostgres(res, "data")
    img = DataLoader.todepth(img, 3)
    print(img:size())
    img = image.crop(img, v["left"], v["top"], v["left"] + v["width"], v["top"] + v["height"])
    img = image.scale(img, 40, 40)
    if v["label"] == "H" then
        --image.save("healthy-" .. v["image_id"] .. "-" .. v["id"] .. ".jpg", img)
    elseif v["label"] == "L" then
        --image.save("leuko-" .. v["image_id"] .. "-" .. v["id"] .. ".jpg", img)
    end
    print('done processing a crop')
end

--local image_width, image_height = 1024, 1024
--local crop1, crop2, crop3, crop4 = {50,50,100,100}, {49,49,100,100}, {100,100,200,200}, {150,150,100,100}
--local pixelsets = {}
--table.insert(pixelsets, pixelset(crop1, image_width, image_height))
--table.insert(pixelsets, pixelset(crop2, image_width, image_height))
--table.insert(pixelsets, pixelset(crop3, image_width, image_height))
--table.insert(pixelsets, pixelset(crop4, image_width, image_height))

--for i = 1, #pixelsets do
--    for j = i + 1, #pixelsets do
--        local sim, iorj = setsimilarity(pixelsets[i], pixelsets[j])
--        print(i, j, sim, iorj)
--    end
--end
