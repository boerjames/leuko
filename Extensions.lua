-- Extensions to existing lua and torch classes

function string.fromhex(str)
    return (str:gsub('..', function (cc)
        return string.char(tonumber(cc, 16))
    end))
end

function string.tohex(str)
    return (str:gsub('.', function (c)
        return string.format('%02X', string.byte(c))
    end))
end

function table.length(t)
    local count = 0
    for _ in pairs(t) do count = count + 1 end
    return count
end
