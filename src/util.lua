--
-- Created by IntelliJ IDEA.
-- User: James
-- Date: 10/8/15
-- Time: 10:07 AM
-- To change this template use File | Settings | File Templates.
--

function fileExists(name)
    local f = io.open(name,'r')
    if f ~= nil then
        io.close(f)
        return true
    else
        return false
    end
end

function osCommand(command)
    local file = assert(io.popen(command))
    local output = file:read('*all')
    file:close()
    return output
end