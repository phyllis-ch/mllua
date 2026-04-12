local M = {}

-- Matrix
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



-- Macros

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



-- Activation Functions

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



-- Neural Network
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

-- Subject to change
function M.header_create(layers, func)
   func = func or {}
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
      if M[func[i]] == nil then
         print("Error: Activation function does not exist")
         os.exit(1)
      end

      if func[i] == 0 then
         -- skip
      else
         M[func[i]](nn["a" .. i])
      end
   end

   return nn["a" .. #func][1][1]
end

return M
