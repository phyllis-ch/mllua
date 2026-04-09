local nn = require("neural")
local mat = require("matrix")

local function forward(Xor)
   -- for i = 1,3 do
   --    mat.dot(Xor["a" .. i], Xor["a" .. i-1], Xor["w" .. i])
   --    mat.sum(Xor["a" .. i], Xor["b" .. i])
   --    mat.relu(Xor["a" .. i])
   -- end

   mat.dot(Xor["a1"], Xor["a0"], Xor["w1"])
   mat.sum(Xor["a1"], Xor["b1"])
   mat.sigmoid(Xor["a1"])

   mat.dot(Xor["a2"], Xor["a1"], Xor["w2"])
   mat.sum(Xor["a2"], Xor["b2"])
   mat.relu(Xor["a2"])


   return Xor["a2"][1][1]
end

local function cost(Xor, ti, to)
   local result = 0.0

   for i = 1, #ti do
      Xor["a0"][1][1] = ti[i][1]
      Xor["a0"][1][2] = ti[i][2]

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
   {0.5, 0.5},
   {0.7, 0.2},
   {1.0, 0.0},
   {0.8, 0.8},
   {-0.5, -0.5},
   {-1.0, 0.0},
}

local to = {
   {1.0},
   {1.0},
   {1.0},
   {0.0},
   {0.0},
   {1.0},
   {0.0},
}

local Xor = nn.init({2, 4, 3, 1})
nn.randomf(Xor, 0, 1)
local Grad = nn.init({2, 4, 3, 1})

local eps = 1e-3
local rate = 1e-1

mat.PRINT(Xor, "a0")
mat.PRINT(Xor, "w1")
print("OK")
for _ = 1, 10000 do
   print("cost = " .. cost(Xor, ti, to))
   finite_diff(Xor, Grad, eps, ti, to)
   train(Xor, Grad, rate)
end

-- Print result
print("---------------------------------")
print("Xor gate: ")

Xor["a0"][1][1] = 0
Xor["a0"][1][2] = 0
print(string.format("%d ^ %d = %f", 0, 0, forward(Xor)))
Xor["a0"][1][1] = 0.8
Xor["a0"][1][2] = 0.8
print(string.format("%f ^ %f = %f", 0.8, 0.8, forward(Xor)))
Xor["a0"][1][1] = -0.5
Xor["a0"][1][2] = -0.5
print(string.format("%f ^ %f = %f", -0.5, -0.5, forward(Xor)))
