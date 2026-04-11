local M = {}

-- TODO: Add an activation function
-- like sigmoid and relu

function M.mat_init(x, y)
   local m = {}
   for i = 1, x do
      m[i] = {}
      for j = 1, y do
         m[i][j] = 0
      end
   end
   return m
end

function M.mat_fill(m, a)
   for i = 1, #m do
      for j = 1, #m[1] do
         m[i][j] = a
      end
   end
end

function M.mat_print(m)
   for i = 1, #m do
      for j = 1, #m[1] do
         io.write("   ")
         io.write(m[i][j])
      end
      io.write("\n")
   end
end

function M.MAT_PRINT(l, m)
   for k, v in pairs(l) do
      if k == m then
         io.write(k .. ": \n")
         M.print(v)
         break
      end
   end
end

function M.mat_randomf(m, low, high)
   for i = 1, #m do
      for j = 1, #m[1] do
         m[i][j] = math.random() * (high - low) + low
      end
   end
end

function M.mat_sum(m, n)
   assert(#m == #n, "Rows are not equal")
   assert(#m[1] == #n[1], "Columns are not equal")
   for i = 1, #m do
      for j = 1, #m[1] do
         m[i][j] = m[i][j] + n[i][j]
      end
   end
end

function M.mat_dot(dst, m, n)
   assert(#m[1] == #n, "Rows and columns are not equal")

   for i = 1, #m do
      for j = 1, #n[1] do
         dst[i][j] = 0
         for k = 1, #m[1] do
            dst[i][j] = dst[i][j] + (m[i][k] * n[k][j])
         end
      end
   end
end

function M.sigmoid(m)
   for i = 1, #m do
      for j = 1, #m[1] do
         m[i][j] = math.exp(m[i][j]) / (1 + math.exp(m[i][j]))
      end
   end
end

function M.relu(m)
   for i = 1, #m do
      for j = 1, #m[1] do
         m[i][j] = math.max(0, m[i][j])
      end
   end
end

return M
