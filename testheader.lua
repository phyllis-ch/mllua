local nua = require("nua")

local function forward(header)
   local func = header["arr_func"]
   local nn = header["nn"]

   for i = 1, #func do
      nua.mat_dot(nn["a" .. i], nn["a" .. i-1], nn["w" .. i])
      nua.mat_sum(nn["a" .. i], nn["b" .. i])
      if func[i] == 0 then
         -- skip
      else
         nua[func[i]](nn["a" .. i])
      end
   end

   return nn["a" .. #func][1][1]
end

local function cost(header, ti, to)
   local nn = header["nn"]
   local result = 0.0

   for i = 1, #ti do
      nn["a0"][1][1] = ti[i][1]
      nn["a0"][1][2] = ti[i][2]

      local y = forward(header)
      local d = y - to[i][1]
      result = result + d*d
   end
   return result / #ti
end

local function finite_diff(header, eps, ti, to)
   local c = cost(header, ti, to)
   local nn = header["nn"]
   local Grad = nua.init(header["arr_layers"])

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            local temp = v[i][j]

            v[i][j] = v[i][j] + eps
            Grad[k][i][j] = (cost(header, ti, to) - c) / eps
            v[i][j] = temp
         end
      end
   end

   return Grad
end

local function train(header, Grad, rate)
   local nn = header["nn"]

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            v[i][j] = v[i][j] - (rate * Grad[k][i][j])
         end
      end
   end
end


-- Main

local ti = {
   {0, 0},
   {0, 1},
   {1, 0},
   {1, 1}
}

local to = {
   {0},
   {1},
   {1},
   {0}
}

local header = nua.header_create({2, 2, 1})
local Xor = header["nn"]
nua.randomf(Xor, 0, 1)

header["arr_func"] = { "sigmoid", "relu" }

local eps = 1e-3
local rate = 1e-1

for _ = 1, 5000 do
   print("cost = " .. cost(header, ti, to))
   local Grad = finite_diff(header, eps, ti, to)
   train(header, Grad, rate)
end

-- Print result
print("---------------------------------")
print("Xor gate: ")
for i = 0, 1 do
   for j = 0, 1 do
      Xor["a0"][1][1] = i
      Xor["a0"][1][2] = j
      print(string.format("%d ^ %d = %f", i, j, forward(header)))
   end
end

-- for i = 1, #header["arr_layers"] do
--    print(header["arr_layers"][i])
-- end
--
-- for i = 1, #header["arr_func"] do
--    print(header["arr_func"][i])
-- end
--
-- nua.print(Xor)
