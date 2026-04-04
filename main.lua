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
local function cost(w)
   local result = 0.0

   for i = 1, data_size do
      local x = train_data[i][1]
      local y = x * w
      local d = y - train_data[i][2]
      result = result + d*d
   end
   result = result / data_size

   return result
end


-- Stir the pile until it works
local w = math.random() * 10                    -- Random float between 1 to 10
local eps = 1e-3
local rate = 1e-3                               -- learning rate to make dcost smaller, to wiggle slowly

-- Wiggle w to the better direction using dcost
-- If cost result becomes lower, then correct direction
print(cost(w))
for i = 1, 500 do
   local dcost = (cost(w + eps) - cost(w)) / eps   -- Derivative function
   w = w - (rate * dcost)
   print(cost(w))
end

-- If w is basically 2, then good
print("------------------")
print(w)
