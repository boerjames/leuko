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

local function imagefrompostgres(data, idx)
    local str = data[idx]:sub(3, data[idx]:len())
    local bytes = torch.ByteTensor(torch.ByteStorage():string(str:fromhex()))
    local img = image.decompressJPG(bytes)
    return img
end

local function loadData(data_path, data_size, all_data, test_percentage, valid_percentage, verbose)
	if verbose then print('Loading Data') end

	local c, h, w = data_size[1], data_size[2], data_size[3]

	local normal = paths.indexdir(paths.concat(data_path, 'normal'))		-- 1
	local leuko	 = paths.indexdir(paths.concat(data_path, 'leuko'))			-- 2

	local num_normal, num_leuko
	if all_data then
		num_normal, num_leuko = normal:size(), leuko:size()
	else
		num_normal, num_leuko = 1000, 1000
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
DataLoader.imagefrompostgres = imagefrompostgres
DataLoader.todepth = todepth

return DataLoader
