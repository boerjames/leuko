require 'dp'
require 'torchx'

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
local image_cursor = conn:execute("select image.id from image")
local num_image_rows = image_cursor:numrows()
local image_res = {}
local crops = {}

for i = 1, 5 do
    image_cursor:fetch(image_res, "a")
    --print('image ' .. image_res["id"])
    print(image_res)

    local eye_cursor = conn:execute("select eye_tag.id, eye_tag.label, eye_tag.top, eye_tag.left, eye_tag.width, eye_tag.height from eye_tag where eye_tag.image_id=" .. image_res["id"])
    local num_eye_tags = eye_cursor:numrows()
    local eye_tags, eye_tags_tmp = {}, {}

    for j = 1, num_eye_tags do
        eye_cursor:fetch(eye_tags_tmp, "a")
        if eye_tags_tmp[""] == "H" or eye_tags_tmp[""] == "L" then
            table.insert(eye_tags, eye_tags_tmp)
        end
    end

    if #eye_tags > 0 then

    end

    collectgarbage()
    print(image_res["id"])
    print(eye_tags)
    print()
end

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
