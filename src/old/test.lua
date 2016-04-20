local image_width, image_height = 200, 200
local set1, set2 = {}, {}

local function fillpixelset(crop, image_width, image_height)
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

    print('s1len',s1len)
    print('s2len',s2len)
    print('ulen',ulen)
    print('ilen',ilen)
    print('overlap',overlap)

    if overlap < 0.5 then return nil end
    if s2len > s1len then
        return s2
    else
        return s1
    end
end

set1 = fillpixelset(set1, {50,50,50,50}, image_width, image_height)
set2 = fillpixelset(set2, {75,75,50,50}, image_width, image_height)
local res = setsimilarity(set1, set2)
if res == nil then print('nil') else print('not nil') end

--th hypero/scripts/export.lua --batteryName 'Neural Network - Mnist' --versionDesc 'Neural Network v1' --metaNames 'hostname' --resultNames 'trainAcc,validAcc,testAcc' --orderBy 'validAcc' --desc
