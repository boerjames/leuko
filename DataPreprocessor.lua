require 'torch'
require 'nn'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

DataPreprocessor = {}

local function rgb2yuv(train, valid, test)

	for i = 1, train.data:size(1) do
		train.data[i] = image.rgb2yuv(train.data[i])
	end

	for i = 1, valid.data:size(1) do
		valid.data[i] = image.rgb2yuv(valid.data[i])
	end

	for i = 1, test.data:size(1) do
		test.data[i] = image.rgb2yuv(test.data[i])
	end

	return train, test
end

local function lcn(train, valid, test, save)
	local normalizationFilter = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))

	-- normalize y locally
	for i = 1, train.data:size(1) do
		local yuv = train.data[i]
		yuv[1] = normalizationFilter(yuv[{{1}}])
		train.data[i] = yuv
	end

	for i = 1, valid.data:size(1) do
		local yuv = valid.data[i]
		yuv[1] = normalizationFilter(yuv[{{1}}])
		valid.data[i] = yuv
	end

	for i = 1, test.data:size(1) do
		local yuv = test.data[i]
		yuv[1] = normalizationFilter(yuv[{{1}}])
		test.data[i] = yuv
	end

	-- normalize u globally
	local mean_u = train.data[{ {}, 2, {}, {} }]:mean()
	local std_u  = train.data[{ {}, 2, {}, {} }]:std()
	train.data[{ {}, 2, {}, {} }]:add(-mean_u)
	train.data[{ {}, 2, {}, {} }]:div(-std_u)
	valid.data[{ {}, 2, {}, {} }]:add(-mean_u)
	valid.data[{ {}, 2, {}, {} }]:div(-std_u)
	test.data[{ {}, 2, {}, {} }]:add(-mean_u)
	test.data[{ {}, 2, {}, {} }]:div(-std_u)

	-- normalize v globally
	local mean_v = train.data[{ {}, 3, {}, {} }]:mean()
	local std_v  = train.data[{ {}, 3, {}, {} }]:std()
	train.data[{ {}, 3, {}, {} }]:add(-mean_v)
	train.data[{ {}, 3, {}, {} }]:div(-std_v)
	valid.data[{ {}, 3, {}, {} }]:add(-mean_v)
	valid.data[{ {}, 3, {}, {} }]:div(-std_v)
	test.data[{ {}, 3, {}, {} }]:add(-mean_v)
	test.data[{ {}, 3, {}, {} }]:div(-std_v)

	if save then
		torch.save('mean_u.t7', mean_u, 'ascii')
		torch.save('std_u.t7', std_u, 'ascii')
		torch.save('mean_v.t7', mean_v, 'ascii')
		torch.save('std_v.t7', std_v, 'ascii')
	end

	return train, valid, test

end

-- crop the given image on the given crop boundary, with some random given variation in translation
function randomTranslation(img, crop_boundary, variation)
    --[[
    img: the torch image to crop

    crop_boundary: coincides with crop_boundary defintion from database
     [1] left boundary
     [2] right boundary
     [3] width of crop
     [4] height of crop

    variation: the number of pixels to randomly vary in translation
    --]]

    local x = crop_boundary[1] + (torch.uniform(0, variation) - variation / 2)
    local y = crop_boundary[2] + (torch.uniform(0, variation) - variation / 2)
    local width = crop_boundary[3]
    local height = crop_boundary[4]

    return image.crop(img, x, y, x + width, y + height)
end

-- crop the given image on the given crop boundary, with some random given variation in scale
function randomScale(img, crop_boundary, variation)
    --[[
    img: the torch image to crop

    crop_boundary: coincides with crop_boundary defintion from database
     [1] left boundary
     [2] right boundary
     [3] width of crop
     [4] height of crop

    variation: the number of pixels to randomly vary in scale
    --]]
    local v = torch.uniform(0, variation) - variation / 2
    local x = crop_boundary[1] + v
    local y = crop_boundary[2] + v
    local width = crop_boundary[3] - 2 * v
    local height = crop_boundary[4] - 2 * v

    return image.crop(img, x, y, x + width, y + height)
end

function randomRotation()
end

function randomFlip()
end

DataPreprocessor.rgb2yuv 			= rgb2yuv
DataPreprocessor.lcn				= lcn
DataPreprocessor.randomTranslation	= randomTranslation
DataPreprocessor.randomScale		= randomScale

return DataPreprocessor
