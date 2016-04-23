require 'dp'
require 'torchx'

--[[how to use]]-- $> th main.lua [flag] [parameter]

--[[command line arguments]]
local cmd = torch.CmdLine()
cmd:text()

--[[database connection settings]]
cmd:option('--host',                'facetag-db',   'the host connect to')
cmd:option('--dbname',              'facetag',      'the db to use')
cmd:option('--user',                'facetag',      'the user to use')
cmd:option('--password',            '',             'the password for the user, do not fill this in, use cmd line')

--[[data parameters]]
cmd:option('--validPercentage',     0.15,           'percentage of data to use for validation')
cmd:option('--testPercentage',      0.15,           'perctage of date to use for testing')
cmd:option('--dataSize',            '{3,40,40}',    'the shape of the input data')

--[[preprocessing]]
cmd:option('--lcn', false, 'apply Yann LeCun Local Contrast Normalization')

--[[verbosity]]
cmd:option('--silent', false, 'dont print anything to stdout')

--[[parse these options]]
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
--local eye_cursor = conn:execute("select eye_tag.id, eye_tag.label, eye_tag.left, eye_tag.top, eye_tag.width, eye_tag.height, eye_tag.image_id from eye_tag")
local image_cursor = conn:execute("select image.id from image")
local num_image_rows = image_cursor:numrows()
local image_res = {}

for i = 1, num_image_rows do
    image_cursor:fetch(image_res, "a")
    print('image ' .. image_res["id"])

    local eye_cursor = conn:execute("select eye_tag.id, eye_tag.label, eye_tag.top, eye_tag.left, eye_tag.width, eye_tag.height from eye_tag where eye_tag.image_id=" .. image_res["id"])
    local num_eye_tags = eye_cursor:numrows()
    local eye_tags, eye_tags_tmp = {}, {}

    for j = 1, num_eye_rows do
        eye_cursor:fetch(eye_tags_tmp, "a")
        table.insert(eye_tags, eye_tags_tmp)
    end

    print(image_res["id"])
    print(eye_tags)
    print()
end
