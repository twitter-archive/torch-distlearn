
local function AllReduceEA(tree, tau, alpha)

   -- Keep track of how many steps each node does per epoch
   local step = 0

   -- Keep track of the center point (also need space for the delta)
   local center,delta,flatParam

   -- Average the parameters according to http://arxiv.org/abs/1412.6651
   local function averageParameters(params)
      -- First time we need to initialize the center point and delta
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

   local function synchronizeCenter(params)
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
      -- Scatter the center point
      tree.scatter(center)
      -- Reset step counter
      step = 0
   end

   return {
      averageParameters = averageParameters,
      synchronizeCenter = synchronizeCenter,
   }
end

return AllReduceEA
