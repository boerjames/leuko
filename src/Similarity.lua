Similarity = {}

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

-- expose desired functions as public
Similarity.cropsimilarity = cropsimilarity

return Similarity
