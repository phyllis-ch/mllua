local train_data = {
   {0.0, 0.0, 0.0},
   {1.0, 0.0, 1.0},
   {0.0, 1.0, 1.0},
   {1.0, 1.0, 0.0},
}
local data_size = #train_data

local function sigmoid(x)
   return math.exp(x) / (1 + math.exp(x))
end

local function forward(Xor, x1, x2)
      local a1 = sigmoid((x1 * Xor.w1) + (x2 * Xor.w3) + Xor.b1)
      local a2 = sigmoid((x1 * Xor.w2) + (x2 * Xor.w4) + Xor.b2)
      return sigmoid((a1 * Xor.w5) + (a2 * Xor.w6) + Xor.b3)
end

local function cost(Xor)
   local result = 0.0

   for i = 1, data_size do
      local x1 = train_data[i][1]
      local x2 = train_data[i][2]

      local y = forward(Xor, x1, x2)
      local d = y - train_data[i][3]
      result = result + d*d
   end
   return result / data_size
end

local function finite_diff(Xor, eps)
   local Grad = {}

   local c = cost(Xor)
   for k, v in pairs(Xor) do
      local temp = v

      Xor[k] = v + eps
      Grad[k] = (cost(Xor) - c) / eps
      Xor[k] = temp
   end

   return Grad
end

local function train(Xor, Grad, rate)
   for k, v in pairs(Xor) do
      Xor[k] = v - (rate * Grad[k])
   end
   return Xor
end


-- Main
local Xor = {
   w1 = math.random(),
   w2 = math.random(),
   b1 = math.random(),

   w3 = math.random(),
   w4 = math.random(),
   b2 = math.random(),

   w5 = math.random(),
   w6 = math.random(),
   b3 = math.random(),
}

local eps = 1e-2
local rate = 1e-2

-- print("w1 = " .. Xor.w1)
-- print("w2 = " .. Xor.w2)
-- print("b1 = " .. Xor.b1)
--
-- print("w3 = " .. Xor.w3)
-- print("w4 = " .. Xor.w4)
-- print("b2 = " .. Xor.b2)
--
-- print("w5 = " .. Xor.w5)
-- print("w6 = " .. Xor.w6)
-- print("b3 = " .. Xor.b3)
print("cost = " .. cost(Xor) .. "\n")

-- print("w1 = " .. Grad.w1)
-- print("w2 = " .. Grad.w2)
-- print("b1 = " .. Grad.b1)
--
-- print("w3 = " .. Grad.w3)
-- print("w4 = " .. Grad.w4)
-- print("b2 = " .. Grad.b2)
--
-- print("w5 = " .. Grad.w5)
-- print("w6 = " .. Grad.w6)
-- print("b3 = " .. Grad.b3 .. "\n")

-- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
-- !!!!You can change this shit!!!!
-- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
local itr = 150000
for _ = 1, itr do
   Grad = finite_diff(Xor, eps)
   Xor = train(Xor, Grad, rate)
   print("cost = " .. cost(Xor))
end
print("\n")

-- Print result
print(itr .. " iterations")
print("---------------------------------")
print("Xor gate: ")
for i = 0, 1 do
   for j = 0, 1 do
      print(string.format("%d ^ %d = %f", i, j, forward(Xor, i, j)))
   end
end
