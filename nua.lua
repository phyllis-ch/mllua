local mat = require("matrix")

local M = {}

function M.init(layers)
   local nn = {}

   nn["a0"] = mat.init(1, layers[1])
   for i = 1, #layers-1 do
      nn["w" .. i] = mat.init(layers[i], layers[i+1])
      nn["b" .. i] = mat.init(1, layers[i+1])
      nn["a" .. i] = mat.init(1, layers[i+1])
   end

   return nn
end

function M.print(nn)
   for k, v in pairs(nn) do
      print(k .. ": ")
      mat.print(v)
   end
end

function M.randomf(nn, low, high)
   for k, _ in pairs(nn) do
      mat.randomf(nn[k], low, high)
   end
end

function M.fill(nn, x)
   for k, _ in pairs(nn) do
      mat.fill(nn[k], x)
   end
end

return M
