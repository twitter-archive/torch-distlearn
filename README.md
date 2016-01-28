DistLearn
=========

Some common distributed learning algorithms built in Torch
with the help of the the ipc library.

AllReduceSGD
------------

Spreads the computation of gradients for mini-batch of items
across N processes. Uses AllReduce to quickly sum the gradients
and distribute the total back out to every process.

```lua
local allReduceSGD = require 'distlearn.AllReduceSGD'(tree)
-- Make sure all the nodes start with the same parameter values
allReduceSGD.synchronizeParameters(params)
for _ = 1,epochs do
   for _ = 1,steps
      -- Compute your gradients as normal
      local grads = computeYourGrads(...)
      -- Sum and normalize them
      allReduceSGD.sumAndNormalizeGradients(grads)
      -- Do your SGD as normal
      SGD(params, grads)
   end
   -- Before validating we should make sure all nodes have
   -- the exact same parameter values
   allReduceSGD.synchronizeParameters(params)
   -- Validate...
end
```

When used in combination with Dataset you can quickly parallelize
the processing of large datasets without a ton of effort. See the
[MNIST example](examples/mnist.lua) for a complete working setup.

AllReduceEA
-----------

We also have a AllReduce based implementation of the Elastic
Averaging algorithm as described in [Deep learning with Elastic Averaging SGD](http://arxiv.org/abs/1412.6651).
Its just as easy to add this to your training script, there
are only two parameters required tau and alpha. Tau is how
many steps to run before averaging the nodes and alpha is
the weight used during the averaging step. You can read
more about [our implementation of AllReduceEA](lua/AllReduceEA.md).

```lua
-- Use a tau of 10 and an alpha of 0.2
local allReduceEA = require 'distlearn.AllReduceEA'(tree, 10, 0.2)
-- Make sure all the nodes start with the same parameter values
allReduceEA.synchronizeParameters(params)
for _ = 1,epochs do
   for _ = 1,steps
      -- Compute your gradients as normal
      local grads = computeYourGrads(...)
      -- Do your SGD as normal
      SGD(params, grads)
      -- Average the params
      allReduceEA.averageParameters(params)
   end
   -- Make sure the center's haven't drifted too far due to
   -- floating point precision error build up
   allReduceEA.synchronizeCenter(params)
   -- Validate...
end
```

See a complete working example of [EA and MNIST](examples/mnist-ea.lua)

License
-------

Licensed under the Apache License, Version 2.0.
[See LICENSE file](LICENSE).
