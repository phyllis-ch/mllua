local nua = require("nua")

-- Apply diffs
local function learn(header, Grad, rate)
   local nn = header["nn"]

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            v[i][j] = v[i][j] - (rate * Grad[k][i][j])
         end
      end
   end
end

local function train(opts)
   local header = opts.header
   local eps = opts.eps
   local rate = opts.rate
   local epoch = opts.epoch
   local step = opts.step
   local td = opts.td
   local stride = opts.stride

   for i = 1, epoch do
      local Grad = nua.finite_diff(header, eps, td, stride)
      learn(header, Grad, rate)
      if i % step == 0 then
         print("cost = " .. nua.mse_cost(header, td, stride))
      end
   end
end



-- Main
local training_data = {
   {0, 0, 0},
   {0, 1, 1},
   {1, 0, 1},
   {1, 1, 0}
}


local training_input = {
   {0, 0},
   {0, 1},
   {1, 0},
   {1, 1}
}

local training_output = {
   {0},
   {1},
   {1},
   {0}
}

-- nua.header_create takes in a table of integers, and an optional array of activation functions
-- Table count will be the layers of neural network 
-- Each element of the table are amount of neurons in each layer
-- First element will always be amount of input
-- Last element will always be amount of output
--
-- nua.header_create creates a header
-- A header consists of
-- 1. array of layers passed as input
-- 2. array of activation functions
-- 3. a pointer to the actual neural network

-- Amount of activation functions should always be one less than length of array of layers
-- As in #func = #layers-1
local header = nua.nn.new({2, 2, 1}, {"sigmoid", "relu"})
-- To make it less crowded, a variable that points to the neural network can be created
local Xor = header["nn"]
nua.nn.randomf(Xor, 0, 1)

-- TODO: A function to interact with the array will probably be better
-- header["arr_func"] = { "sigmoid", "relu" }

train({
   header = header,
   eps = 1e-3,        -- epsilon (small number)
   rate = 1e-1,              -- learning rate
   epoch = 5000,             -- amount of training
   step = 500,              -- amount of steps before printing out cost
   td = training_data,
   stride = 2
})

-- Print result
print("---------------------------------")
print("Xor gate: ")
for i = 0, 1 do
   for j = 0, 1 do
      Xor["a0"][1][1] = i
      Xor["a0"][1][2] = j
      print(string.format("%d ^ %d = %f", i, j, nua.nn.forward(header)[1][1]))
   end
end
