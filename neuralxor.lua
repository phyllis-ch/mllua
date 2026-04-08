local nn = require("neural")
local mat = require("matrix")

local function forward(Xor)
   mat.dot(Xor["a1"], Xor["x"], Xor["w1"])
   mat.sum(Xor["a1"], Xor["b1"])
   mat.sigmoid(Xor["a1"])

   mat.dot(Xor["a2"], Xor["a1"], Xor["w2"])
   mat.sum(Xor["a2"], Xor["b2"])
   mat.sigmoid(Xor["a2"])

   return Xor["a2"][1][1]
end

local function cost(Xor, ti, to)
   local result = 0.0

   for i = 1, #ti do
      Xor["x"][1][1] = ti[i][1]
      Xor["x"][1][2] = ti[i][2]

      local y = forward(Xor)
      local d = y - to[i][1]
      result = result + d*d
   end
   return result / #ti
end

local function finite_diff(Xor, Grad, eps, ti, to)
   local c = cost(Xor, ti, to)

   for k, v in pairs(Xor) do
      for i = 1, #v do
         for j = 1, #v[1] do
            local temp = v[i][j]

            v[i][j] = v[i][j] + eps
            Grad[k][i][j] = (cost(Xor, ti, to) - c) / eps
            v[i][j] = temp
         end
      end
   end
end

local function train(Xor, Grad, rate)
   for k, v in pairs(Xor) do
      for i = 1, #v do
         for j = 1, #v[1] do
            v[i][j] = v[i][j] - (rate * Grad[k][i][j])
         end
      end
   end
end


-- Main
local ti = {
   {0.0, 0.0},
   {0.0, 1.0},
   {1.0, 0.0},
   {1.0, 1.0},
}

local to = {
   {0.0},
   {1.0},
   {1.0},
   {0.0},
}

local Xor = nn.init({2, 2, 1})
nn.randomf(Xor, 0, 1)
local Grad = nn.init({2, 2, 1})

local eps = 1e-1
local rate = 1e-1

for _ = 1, 100000 do
   print("cost = " .. cost(Xor, ti, to))
   finite_diff(Xor, Grad, eps, ti, to)
   train(Xor, Grad, rate)
end

-- Print result
print("---------------------------------")
print("Xor gate: ")
for i = 0, 1 do
   for j = 0, 1 do
      Xor["x"][1][1] = i
      Xor["x"][1][2] = j
      print(string.format("%d ^ %d = %f", i, j, forward(Xor)))
   end
end
