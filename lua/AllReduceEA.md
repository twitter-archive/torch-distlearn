
Elastic Averaging using AllReduce
=================================

The published algorithm for Elastic Averaging can be found in in the
[Deep learning with Elastic Averaging SGD](http://arxiv.org/abs/1412.6651)
paper.  We took this exact math and transformed to run as a single AllReduce
operation. In order to do so each node needs to have its own copy
of the center point. When the algorithm begins all nodes have the same
params and the same center point.

```lua
   -- Is it time to do an elastic average?
   if step % tau == 0 then
      -- Compute this node's elastic difference
      delta = (params - center) * alpha
      -- Move this node towards the center point
      params = params - delta
      -- Sum the elastic differences from all nodes
      allDeltas = allReduce(delta, function(a, b) return a + b end)
      -- Move the center towards all the nodes
      center = center + allDeltas
   end
```

This is mathematically equivalent to the paper's algorithm but has
the advantage of only requiring T*log2(N) time to run,
where N is the number of nodes involved in training, and T is
defined as the time it takes to transfer params between any two nodes
in the network.
