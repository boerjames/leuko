require 'torch'
require 'nn'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')

DataShuffler = {}

local function shuffle(examples)
	local size = examples.data:size(1)
	local shuffle = torch.randperm(size):long()
	examples.data = examples.data:index(1,shuffle)
	examples.labels = examples.labels:index(1,shuffle)
	return examples
end

DataShuffler.shuffle = shuffle

return DataShuffler