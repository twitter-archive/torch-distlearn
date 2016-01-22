
local function AllReduceEA(tree, tau, alpha)

   -- Keep track of how many steps each node does per epoch
   local step = 0

   -- Keep track of the center point (also need space for the delta)
   local center,delta,flatParam

   -- Clone the parameters to use a center point
   local function oneTimeInit(params)
      if not center then
         center = { }
         delta = { }
         flatParam = { }
         tree.walkTable(params, function(param)
            table.insert(center, param:clone())
            table.insert(delta, param:clone())
            table.insert(flatParam, param)
         end)
      end
   end

   -- Average the parameters according to http://arxiv.org/abs/1412.6651
   local function averageParameters(params)
      -- First time we need to initialize the center point and delta
      oneTimeInit(params)
      -- This node contributed to this step
      step = step + 1
      -- If its time to run an average
      if step % tau == 0 then
         -- Compute our elastic difference (delta)
         -- and move this node towards the center point
         local i = 1
         tree.walkTable(params, function(param)
            delta[i]:add(param, -1, center[i]):mul(alpha)
            param:add(-1, delta[i])
            i = i + 1
         end)
         -- AllReduce the elastic differences
         tree.allReduce(delta, function(a, b) return a:add(b) end)
         -- Move the center point towards the nodes
         for i = 1,#center do
            center[i]:add(delta[i])
         end
      end
   end

   -- Do some fanciness to get all the nodes to the same point
   local function handleUnevenSteps(params)
      -- Only need to synchronize nodes if we have done at least one step
      if step > 0 then
         -- Wow, this is expressed terribly
         -- Do one final all reduce to get all the nodes in sync
         for i = 1,#delta do
            delta[i]:fill(0)
         end
         tree.allReduce(delta,
            function(a, b) return a:add(b) end,
            function(_, i)
               -- Move the center point towards the nodes
               center[i]:add(delta[i])
               -- Compute our elastic difference (delta)
               -- and move this node towards the center point
               delta[i]:add(flatParam[i], -1, center[i]):mul(alpha)
               flatParam[i]:add(-1, delta[i])
               return delta[i]
            end)
         -- Reset step counter
         step = 0
      end
   end

   -- Ensure the same exact center point is on every node
   -- Call at the end of epoch (or at any point you desire)
   -- Over time the center points will drift a bit due to floating point error accumulation
   local function synchronizeCenter(params)
      -- First time we need to initialize the center point and delta
      oneTimeInit(params)
      -- Handle uneven # of steps per node
      handleUnevenSteps(params)
      -- Scatter the center point
      tree.scatter(center)
   end

   -- At any point in time you can force the same parameters on all nodes
   local function synchronizeParameters(params)
      -- First time we need to initialize the center point and delta
      oneTimeInit(params)
      -- Handle uneven # of steps per node
      handleUnevenSteps(params)
      -- Scatter the parameters
      tree.scatter(params)
      -- Reset the center to the parameters
      local i = 1
      tree.walkTable(params, function(param)
         center[i]:copy(param)
         i = i + 1
      end)
   end

   return {
      averageParameters = averageParameters,
      synchronizeCenter = synchronizeCenter,
      synchronizeParameters = synchronizeParameters,
   }
end

return AllReduceEA
