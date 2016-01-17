
-- The all reduce version of SGD's gradient compute
-- that splits a mini-batch across many nodes
local function AllReduceSGD(tree)

   -- Keep track of how many steps each node does per epoch
   local stepsPerNode = torch.LongTensor(tree.numNodes):fill(0)

   -- Sum the gradients of all nodes
   local function sumGradients(grads)
      -- All reduce and sum the gradients
      local _,n = tree.allReduce(grads, function(a, b) return a:add(b) end)
      -- This node contributed to this step
      stepsPerNode[tree.nodeIndex] = stepsPerNode[tree.nodeIndex] + 1
   end

   -- Sum and normalize the gradients of all nodes
   local function sumAndNormalizeGradients(grads)
      -- All reduce and sum the gradients
      local _,n = tree.allReduce(grads, function(a, b) return a:add(b) end)
      -- Normalize them by the # of nodes that contributed
      -- Not all nodes contribute to every step due to uneven partitioning of data
      tree.walkTable(grads, function(grad)
         grad:div(n)
      end)
      -- This node contributed to this step
      stepsPerNode[tree.nodeIndex] = stepsPerNode[tree.nodeIndex] + 1
   end

   -- Get the same parameters on all nodes after a training epoch
   local function synchronizeParameters(params)
      -- Do one final all reduce to get all the nodes in sync
      tree.allReduce(nil, function(a, b) return a:add(b) end, function(a) return a:fill(0) end)
      -- All reduce the # of steps per node
      tree.allReduce(stepsPerNode, function(a, b) return a:add(b) end)
      -- Which node had the greatest # of steps?
      local _,indicies = stepsPerNode:sort()
      -- Zero out our parameters if we aren't the longest
      if tree.nodeIndex ~= indicies[tree.numNodes] then
         tree.walkTable(params, function(param) return param:fill(0) end)
      end
      -- All reduce the params, this puts the winning node's params on all nodes
      tree.allReduce(params, function(a, b) return a:add(b) end)
      -- Reset the steps counter for the next loop of training
      stepsPerNode:fill(0)
   end

   return {
      sumGradients = sumGradients,
      sumAndNormalizeGradients = sumAndNormalizeGradients,
      synchronizeParameters = synchronizeParameters,
   }
end

return AllReduceSGD
