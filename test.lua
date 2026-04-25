local nua = require("nua")

local function mse_cost(header, train_dataset, stride)
   local output_count = #nua.nn.forward(header)[1]
   local nn = header["nn"]
   local result = 0.0

   for i = 1, #train_dataset do
      for j = 1, stride do
         nn["a0"][1][j] = train_dataset[i][j]
      end

      for j = 1, output_count do
         local y = nua.nn.forward(header)[1][j]
         local d = y - train_dataset[i][stride+j]
         result = result + d*d
      end
   end
   return result/ #train_dataset
end

local function finite_diff(header, eps, train_dataset, stride)
   local c = mse_cost(header, train_dataset, stride)
   local nn = header["nn"]
   local Diffs = nua.nn.init(header["arr_layers"])

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            local temp = v[i][j]

            v[i][j] = v[i][j] + eps
            Diffs[k][i][j] = (mse_cost(header, train_dataset, stride) - c) / eps
            v[i][j] = temp
         end
      end
   end

   return Diffs
end

-- gradient descent
-- w_new = w_old - rate * dL/dw
-- b_new = b_old - rate * dL/db

-- x = input, y = expected output, y_act = actual output
-- z = w * input + b
-- dL/dw = dL/dy_act * dy_act/dz * dz/dw
-- dL/db = dL/dy_act * dy_act/dz * dz/db

-- dL/dy_act = 2 * (y_act - y)
-- dy_act/dz = 1     (linear activation)
-- dz/dw = x
-- dz/db = 1
function backprop(header, train_dataset, stride, rate)
   local nn = header["nn"]

   nn["a0"][1][1] = train_dataset[1][1]
   local output = nua.nn.forward(header)[1][1]

   -- dL/dy_act
   local dL_da1 = 2 * (output - train_dataset[1][2])

   local da1_dz = 1 -- linear

   local dz_dw = train_dataset[1][1]

   local dz_db = 1

   local dL_dw = dL_da1 * da1_dz * dz_dw
   local dL_db = dL_da1 * da1_dz * dz_db

   nn["w1"][1][1] = nn["w1"][1][1] - rate * dL_dw
   nn["b1"][1][1] = nn["b1"][1][1] - rate * dL_db
end

local function better_backprop(header, train_dataset, stride, rate)
   local nn = header["nn"]
   local layers = header["arr_layers"]

   for i = 1, #train_dataset do
      for j = 1, #nn["a0"][1] do
         nn["a0"][1][j] = train_dataset[i][j]
      end
      local output = nua.nn.forward(header)

      local diff = {}
      for j = 1, #layers-1 do
         diff["diff_cost_per_a"] = 2 * (output[1][1] - train_dataset[i][stride+1])

         -- Will need to perform check to see if is act_function or linear
         diff["diff_a_per_z"] = 1 -- linear

         diff["diff_z_per_w"] = train_dataset[i][1]

         diff["diff_z_per_b"] = 1

         diff["diff_cost_per_w"] = diff["diff_cost_per_a"]*diff["diff_a_per_z"]*diff["diff_z_per_w"]
         diff["diff_cost_per_b"] = diff["diff_cost_per_a"]*diff["diff_a_per_z"]*diff["diff_z_per_b"]
      end

      nn["w1"][1][1] = nn["w1"][1][1] - rate * diff["diff_cost_per_w"]
      nn["b1"][1][1] = nn["b1"][1][1] - rate * diff["diff_cost_per_b"]
   end
end

local function apply_diff(header, Diffs, rate)
   local nn = header["nn"]

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            v[i][j] = v[i][j] - (rate * Diffs[k][i][j])
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
      local Diffs = finite_diff(header, eps, td, stride)
      apply_diff(header, Diffs, rate)
      if i % step == 0 then
         print("cost = " .. mse_cost(header, td, stride))
      end
   end
end


-- Main
local td = {
   {0, 0},
   {1, 2},
   {2, 4},
   {3, 6},
   {4, 8},
}

local nn_h = nua.nn.new({1, 1})
local nn = nn_h["nn"]
nua.nn.randomf(nn, 0, 1)

nua.nn.print(nn)
print("cost = " .. mse_cost(nn_h, td, 1))
for _ = 1, 1000 do
   better_backprop(nn_h, td, 1, 1e-3)
end
nua.nn.print(nn)
print("cost = " .. mse_cost(nn_h, td, 1))

nn["a0"][1][1] = 1
local output = nua.nn.forward(nn_h)
print(string.format("%d * 2 = %f", nn["a0"][1][1], output[1][1]))
os.exit(1)

train({
   header = nn_h,
   td = td,
   eps = 1e-3,    -- epsilon
   rate = 1e-4,   -- learning rate
   epoch = 8000,  -- amount of training
   step = 500,    -- amount of steps before printing out cost
   stride = 1
})

-- Print result
print("---------------------------------")
for i = 0, 10 do
   nn["a0"][1][1] = i
   local output = nua.nn.forward(nn_h)
   print(string.format("%d * 2 = %f", nn["a0"][1][1], output[1][1]))
end
