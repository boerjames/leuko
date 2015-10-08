--
-- Created by IntelliJ IDEA.
-- User: James
-- Date: 10/8/15
-- Time: 9:44 AM
-- To change this template use File | Settings | File Templates.
--

function addTensors(a, b)
    return a:clone():add(b)
end

function normalizeData(trainset)
    -- store the mean, to normalize the test set in the future
    local mean = {}
    -- store the standard-deviation for the future
    local stdv  = {}

    -- over each image channel
    for i=1,3 do
        -- mean estimation and subtraction
        mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean()
        print('Channel ' .. i .. ', Mean: ' .. mean[i])
        trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i])

        -- stdv estimation and scaling
        stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std()
        print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
        trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i])
    end

    return mean, stdv
end