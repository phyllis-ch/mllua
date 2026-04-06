local train_data = {
   {0.0, 0.0},
   {1.0, 2.0},
   {2.0, 4.0},
   {3.0, 6.0},
   {4.0, 8.0},
}
local data_size = #train_data

-- Cost function shows average how far from expected value
-- Cost function returns result
-- Lower result is better
local function cost(w, b)
   local result = 0.0

   for i = 1, data_size do
      local x = train_data[i][1]
      local y = (x * w) + b
      local d = y - train_data[i][2]
      result = result + d*d
   end
   result = result / data_size

   return result
end


-- Stir the pile until it works
local w = math.random() * 10                    -- Random float between 1 to 10
local b = math.random() * 5                     -- bias
local eps = 1e-3
local rate = 1e-2                               -- learning rate to make dcost smaller, to wiggle slowly

-- Wiggle w to the better direction using dcost
-- If cost result becomes lower, then correct direction
print(cost(w, b))
for _ = 1, 1000 do
   local dw = (cost(w + eps, b) - cost(w, b)) / eps   -- Derivative function
   local db = (cost(w, b + eps) - cost(w, b)) / eps   -- Derivative function
   w = w - (rate * dw)
   b = b - (rate * db)
   print("cost: " .. cost(w, b) .. " w: " .. w .. " b: " .. b)
end

-- If w is basically 2, then good
print("--------------------------")
print("cost: " .. cost(w, b) .. " w: " .. w .. " b: " .. b)

-- Print result
for i = 0, 20, 2 do
      print(string.format("%d * w = %f", i, i*w+b))
end
