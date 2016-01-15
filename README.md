DistLearn
=========

Some common distributed learning algorithms built in Torch
with the help of the the parallel library.

AllReduceSGD
------------

Spreads the computation of gradients for mini-batch of items
across N processes. Uses AllReduce to quickly sum the gradients
and distribute the total back out to every process.

```lua
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

License
-------

Licensed under the Apache License, Version 2.0.
[See LICENSE file](LICENSE).
