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
   -- Build a tree of processes

   -- Compute your gradients as normal
   local grads = computeYourGrads(...)
   -- Sum and normalize them
   allReduceSGD.sumAndNormalizeGradients(grads)

```

When used in combination with Dataset you can quickly parallelize
the processing of large datasets without a ton of effort.

```lua

```

License
-------

Licensed under the Apache License, Version 2.0.
[See LICENSE file](LICENSE).
