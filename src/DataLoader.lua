require 'torch'
require 'image'
require 'torchx'
require 'dp'

require './Extensions.lua'

torch.setdefaulttensortype('torch.FloatTensor')

DataLoader = {}

local function todepth(img, depth)
   if depth and depth == 1 then
      if img:nDimension() == 2 then
          -- all good, do nothing
      elseif img:size(1) == 3 or img:size(1) == 4 then
	 img = image.rgb2y(img:narrow(1,1,3))[1]
      elseif img:size(1) == 2 then
         img = img:narrow(1,1,1)
      elseif img:size(1) ~= 1 then
         dok.error('image loaded has wrong #channels', 'image.todepth')
      end
   elseif depth and depth == 3 then
      local chan = img:size(1)
      if chan == 3 then
          -- all good, do nothing
      elseif img:nDimension() == 2 then
         local imgrgb = img.new(3, img:size(1), img:size(2))
         imgrgb:select(1, 1):copy(img)
         imgrgb:select(1, 2):copy(img)
         imgrgb:select(1, 3):copy(img)
         img = imgrgb
      elseif chan == 4 then
         img = img:narrow(1,1,3)
      elseif chan == 1 then
         local imgrgb = img.new(3, img:size(2), img:size(3))
         imgrgb:select(1, 1):copy(img)
         imgrgb:select(1, 2):copy(img)
         imgrgb:select(1, 3):copy(img)
         img = imgrgb
      else
         dok.error('image loaded has wrong #channels', 'image.todepth')
      end
   end
   return img
end

local function imagefromstring(rawstr)
    local str = rawstr:sub(3, rawstr:len())
    local bytes = torch.ByteTensor(torch.ByteStorage():string(str:fromhex()))
    --local img = image.decompressJPG(bytes)
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
    local r = math.random(1, dim / 2)
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

local function loadData(data_path, data_size, equal_representation, test_percentage, valid_percentage, verbose)
	if verbose then print('Loading Data') end

	local c, h, w = data_size[1], data_size[2], data_size[3]

	local normal = paths.indexdir(paths.concat(data_path, 'normal'))		-- 1
	local leuko	 = paths.indexdir(paths.concat(data_path, 'leuko'))			-- 2

	local num_normal, num_leuko
	if equal_representation then
		num_normal = math.min(normal:size(), leuko:size())
                num_leuko = num_normal
	else
		num_normal, num_leuko = normal:size(), leuko:size()
	end

	local size = num_normal + num_leuko
	if verbose then
		print('Normal examples', num_normal)
		print('Leuko examples', num_leuko)
	end

	local shuffle	= torch.randperm(size)
	local input		= torch.FloatTensor(size, c, h, w)
	local target	= torch.IntTensor(size)

	-- 1. load images into input and target tensors
	for i = 1, num_normal do
		if verbose then print('Normal examples loaded:', i, "/", num_normal) end

		local img = image.load(normal:filename(i))
		img = image.scale(img, h, w)
		img = todepth(img, 3)

		local index = shuffle[i]
		input[index]:copy(img)
		target[index] = 1
		collectgarbage()
	end

	for i = 1, num_leuko do
		if verbose then print('Leuko examples loaded:', i, "/", num_leuko) end

		local img = image.load(leuko:filename(i))
		img = image.scale(img, h, w)
		img = todepth(img, 3)

		local index = shuffle[i + num_normal]
		input[index]:copy(img)
		target[index] = 2
		collectgarbage()
	end

	-- 2. divide into train, test, and valid sets
	local num_valid = math.floor(size * valid_percentage)
	local num_test  = math.floor(size * test_percentage)
	local num_train = size - num_valid - num_test

	-- 3. wrap into dp.View
	local train_input  = dp.ImageView('bchw', input:narrow(1, 1, num_train))
	local train_target = dp.ClassView('b', target:narrow(1, 1, num_train))
	local test_input   = dp.ImageView('bchw', input:narrow(1, num_train + 1, num_test))
	local test_target  = dp.ClassView('b', target:narrow(1, num_train + 1, num_test))
	local valid_input  = dp.ImageView('bchw', input:narrow(1, num_train + num_test + 1, num_valid))
	local valid_target = dp.ClassView('b', target:narrow(1, num_train + num_test + 1, num_valid))

	train_target:setClasses({'normal', 'leuko'})
	test_target:setClasses({'normal', 'leuko'})
	valid_target:setClasses({'normal', 'leuko'})

	-- 4. wrap dp.View into dp.DataSet
	local train = dp.DataSet{inputs=train_input, targets=train_target, which_set='train'}
	local test  = dp.DataSet{inputs=test_input, targets=test_target, which_set='test'}
	local valid = dp.DataSet{inputs=valid_input, targets=valid_target, which_set='valid'}

	-- 4. wrap dp.DataSet into dp.DataSource
	local ds = dp.DataSource{train_set=train, test_set=test, valid_set=valid}
	ds:classes{'normal', 'leuko'}

	if verbose then print('Done Loading Data!') end

	return ds
end

-- expose desired functions as public
DataLoader.loadData = loadData
DataLoader.imagefromstring = imagefromstring
DataLoader.todepth = todepth
DataLoader.outercrop = outercrop
DataLoader.augment = augment

return DataLoader
