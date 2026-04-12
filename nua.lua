local M = {}

-- Matrix

function M.mat_init(x, y)
   local mat = {}
   for i = 1, x do
      mat[i] = {}
      for j = 1, y do
         mat[i][j] = 0
      end
   end
   return mat
end

function M.mat_fill(mat, num)
   for i = 1, #mat do
      for j = 1, #mat[1] do
         mat[i][j] = num
      end
   end
end

function M.mat_print(mat)
   for i = 1, #mat do
      for j = 1, #mat[1] do
         io.write("   ")
         io.write(mat[i][j])
      end
      io.write("\n")
   end
end

function M.mat_randomf(mat, low, high)
   for i = 1, #mat do
      for j = 1, #mat[1] do
         mat[i][j] = math.random() * (high - low) + low
      end
   end
end

function M.mat_sum(dst, mat)
   assert(#dst == #mat, "Rows of destination matrix and source matrix are not equal")
   assert(#dst[1] == #mat[1], "Columns of destination matrix and source matrix are not equal")

   for i = 1, #dst do
      for j = 1, #dst[1] do
         dst[i][j] = dst[i][j] + mat[i][j]
      end
   end
end

function M.mat_dot(dst, mat1, mat2)
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
function M.PRINT(nn, matrix)
   for k, v in pairs(nn) do
      if k == matrix then
         io.write(k .. ": \n")
         M.mat_print(v)
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

function M.init(layers)
   local nn = {}

   nn["a0"] = M.mat_init(1, layers[1])
   for i = 1, #layers-1 do
      nn["w" .. i] = M.mat_init(layers[i], layers[i+1])
      nn["b" .. i] = M.mat_init(1, layers[i+1])
      nn["a" .. i] = M.mat_init(1, layers[i+1])
   end

   return nn
end

function M.header_create(layers)
   local header = {}

   header["arr_layers"] = layers
   header["arr_func"] = {}

   for i = 1, #header["arr_layers"]-1 do
      header["arr_func"][i] = 0
   end

   header["nn"] = M.init(header["arr_layers"])

   return header
end

function M.print(nn)
   for k, v in pairs(nn) do
      print(k .. ": ")
      M.mat_print(v)
   end
end

function M.randomf(nn, low, high)
   for k, _ in pairs(nn) do
      M.mat_randomf(nn[k], low, high)
   end
end

function M.fill(nn, x)
   for k, _ in pairs(nn) do
      M.mat_fill(nn[k], x)
   end
end

return M
