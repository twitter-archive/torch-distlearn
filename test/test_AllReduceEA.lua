local test = require 'regress'
local parallel = require 'libparallel'

test {
   testAllReduceEA = function()
      local function theTest(numEpochs, nodeIndex, numNodes, server, client, port)
         local tree = require 'parallel.Tree'(nodeIndex, numNodes, 2, server, client, '127.0.0.1', port)
         local allReduceEA = require 'distlearn.AllReduceEA'(tree, 3, 0.4)
         local params = { torch.Tensor(7):fill(0) }
         local slowit = 1
         for epoch = 1,5 do
            local steps = math.random(45, 53)
            for step = 1,steps do
               params[1]:add(torch.randn(7):div(slowit)) -- wander, no idea
               allReduceEA.averageParameters(params)
               slowit = slowit * 2
            end
            allReduceEA.synchronizeCenter(params)
         end
         return params
      end
      for z = 1,10 do
         local numNodes = math.pow(2, math.random(1, 3))
         local numEpochs = math.random(13, 27)
         local server, port = parallel.server('127.0.0.1')
         local workers = parallel.map(numNodes - 1, function(numEpochs, numNodes, port, theTest, mapid)
            local parallel = require 'libparallel'
            local client = parallel.client('127.0.0.1', port)
            local ret = theTest(numEpochs, mapid + 1, numNodes, nil, client)
            client:close()
            return ret
         end, numEpochs, numNodes, port, theTest)
         local r1 = theTest(numEpochs, 1, numNodes, server, nil, port)
         local ret = { workers:join() }
         server:close()
         for i = 1,numNodes-1 do
            local delta = torch.abs(r1[1] - ret[i][1]):max()
            assert(delta < 1e-6, 'all nodes should be really close together '..delta)
         end
      end
   end,
}
