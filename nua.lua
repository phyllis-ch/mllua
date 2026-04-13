-- Version 0.1
-- TODO: Add stuff like save_param and load_param
-- Add a function to interact with header
-- make sure order in mat_print and nn_print
-- restructure submodules (maybe expand to a directory)
-- Abstract cost function, finite_diff, learn, and train function
-- make some functions take in a table

local M = {}

-- Sections
--
-- 1. Matrix
-- 2. Matrix Macros
-- 3. Activation Functions
-- 4. Neural Network
-- 5. Training



-- 1. Matrix
M.mat = {}

function M.mat.init(x, y)
   local mat = {}
   for i = 1, x do
      mat[i] = {}
      for j = 1, y do
         mat[i][j] = 0
      end
   end
   return mat
end

function M.mat.fill(mat, num)
   for i = 1, #mat do
      for j = 1, #mat[1] do
         mat[i][j] = num
      end
   end
end

function M.mat.print(mat)
   for i = 1, #mat do
      for j = 1, #mat[1] do
         io.write("   ")
         io.write(mat[i][j])
      end
      io.write("\n")
   end
end

function M.mat.randomf(mat, low, high)
   for i = 1, #mat do
      for j = 1, #mat[1] do
         mat[i][j] = math.random() * (high - low) + low
      end
   end
end

function M.mat.sum(dst, mat)
   assert(#dst == #mat, "Rows of destination matrix and source matrix are not equal")
   assert(#dst[1] == #mat[1], "Columns of destination matrix and source matrix are not equal")

   for i = 1, #dst do
      for j = 1, #dst[1] do
         dst[i][j] = dst[i][j] + mat[i][j]
      end
   end
end

function M.mat.dot(dst, mat1, mat2)
   assert(#mat1[1] == #mat2, "Rows of first matrix and columns of second matrix are not equal")
   assert(#dst == #mat1, "Rows of destination matrix and first matrix are not equal")
   assert(#dst[1] == #mat2[1], "Columns of destination matrix and second matrix are not equal")

   for i = 1, #mat1 do
      for j = 1, #mat2[1] do
         dst[i][j] = 0
         for k = 1, #mat1[1] do
            dst[i][j] = dst[i][j] + (mat1[i][k] * mat2[k][j])
         end
      end
   end
end



-- 2. Matrix Macros

-- Macro for mat_print
-- variable matrix has to be a string because Lua
function M.mat.PRINT(nn, mat)
   for k, v in pairs(nn) do
      if k == mat then
         io.write(k .. ": \n")
         M.mat.print(v)
         break
      end
   end
end



-- 3. Activation Functions

function M.sigmoid(mat)
   for i = 1, #mat do
      for j = 1, #mat[1] do
         mat[i][j] = math.exp(mat[i][j]) / (1 + math.exp(mat[i][j]))
      end
   end
end

function M.relu(mat)
   for i = 1, #mat do
      for j = 1, #mat[1] do
         mat[i][j] = math.max(0, mat[i][j])
      end
   end
end



-- 4. Neural Network
M.nn = {}

function M.nn.init(layers)
   local nn = {}

   nn["a0"] = M.mat.init(1, layers[1])
   for i = 1, #layers-1 do
      nn["w" .. i] = M.mat.init(layers[i], layers[i+1])
      nn["b" .. i] = M.mat.init(1, layers[i+1])
      nn["a" .. i] = M.mat.init(1, layers[i+1])
   end

   return nn
end

function M.nn.new(layers, func)
   func = func or {}
   if #func >= #layers then
      error("Error: Too much activation functions", 2)
   end
   local header = {}

   header["arr_layers"] = layers
   header["arr_func"] = {}

   for i = 1, #header["arr_layers"]-1 do
      header["arr_func"][i] = func[i] or 0
   end

   header["nn"] = M.nn.init(header["arr_layers"])

   return header
end

function M.nn.print(nn)
   for k, v in pairs(nn) do
      print(k .. ": ")
      M.mat.print(v)
   end
end

function M.nn.randomf(nn, low, high)
   for k, _ in pairs(nn) do
      M.mat.randomf(nn[k], low, high)
   end
end

function M.nn.fill(nn, x)
   for k, _ in pairs(nn) do
      M.mat.fill(nn[k], x)
   end
end

function M.nn.forward(header)
   local func = header["arr_func"]
   local nn = header["nn"]

   for i = 1, #func do
      M.mat.dot(nn["a" .. i], nn["a" .. i-1], nn["w" .. i])
      M.mat.sum(nn["a" .. i], nn["b" .. i])
      if func[i] == 0 then
         -- skip
      elseif M[func[i]] == nil then
         error("Error: Activation function does not exist", 2)
      else
         M[func[i]](nn["a" .. i])
      end
   end

   return nn["a" .. #func]
end



-- 5. Training

function M.mse_cost(header, train_dataset, stride)
   local output_count = #M.nn.forward(header)[1]
   local nn = header["nn"]
   local result = 0.0

   for i = 1, #train_dataset do
      for j = 1, stride do
         nn["a0"][1][j] = train_dataset[i][j]
      end

      local y = {}
      local d = {}

      for j = 1, output_count do
         y[j] = M.nn.forward(header)[1][j]
         d[j] = y[j] - train_dataset[i][stride+j]
         result = result + d[j]*d[j]
      end
   end
   return result / output_count / #train_dataset
end

function M.finite_diff(header, eps, td, stride)
   local c = M.mse_cost(header, td, stride)
   local nn = header["nn"]
   local Gradients = M.nn.init(header["arr_layers"])

   for k, v in pairs(nn) do
      for i = 1, #v do
         for j = 1, #v[1] do
            local temp = v[i][j]

            v[i][j] = v[i][j] + eps
            Gradients[k][i][j] = (M.mse_cost(header, td, stride) - c) / eps
            v[i][j] = temp
         end
      end
   end

   return Gradients
end

return M
