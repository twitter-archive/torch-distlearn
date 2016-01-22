local opt = lapp [[
Train a CNN classifier on MNIST using AllReduceSGD.

   --nodeIndex         (default 1)
   --numNodes          (default 1)
]]

-- Requires
local grad = require 'autograd'
local util = require 'autograd.util'
local lossFuns = require 'autograd.loss'
local optim = require 'optim'
local Dataset = require 'dataset.Dataset'

-- Build the AllReduce tree
local tree = require 'ipc.LocalhostTree'(opt.nodeIndex, opt.numNodes)
local allReduceSGD = require 'distlearn.AllReduceSGD'(tree)

-- Print only in instance 1!
if opt.nodeIndex > 1 then
   xlua.progress = function() end
   print = function() end
end

-- Load the MNIST dataset
local trainingDataset = Dataset('http://d3jod65ytittfm.cloudfront.net/dataset/mnist/train.t7', {
   partition = opt.nodeIndex,
   partitions = opt.numNodes,
})

local getTrainingBatch, numTrainingBatches = trainingDataset.sampledBatcher({
   samplerKind = 'permutation',
   batchSize = 1,
   inputDims = { 1024 },
   verbose = true,
   processor = function(res, processorOpt, input)
      input:copy(res:view(1, 1024))
      return true
   end,
})

-- Load in MNIST
local classes = {'0','1','2','3','4','5','6','7','8','9'}
local confusionMatrix = optim.ConfusionMatrix(classes)

-- Ensure params are equal on all nodes
torch.manualSeed(0)

-- What model to train:
local predict,f,params

-- for CNNs, we rely on efficient nn-provided primitives:
local reshape = grad.nn.Reshape(1,32,32)

local conv1, acts1, pool1, conv2, acts2, pool2, flatten, linear
local params = {}
conv1, params.conv1 = grad.nn.SpatialConvolutionMM(1, 16, 5, 5)
acts1 = grad.nn.Tanh()
pool1 = grad.nn.SpatialMaxPooling(2, 2, 2, 2)

conv2, params.conv2 = grad.nn.SpatialConvolutionMM(16, 16, 5, 5)
acts2 = grad.nn.Tanh()
pool2, params.pool2 = grad.nn.SpatialMaxPooling(2, 2, 2, 2)

flatten = grad.nn.Reshape(16*5*5)
linear,params.linear = grad.nn.Linear(16*5*5, 10)

-- Cast the parameters
params = grad.util.cast(params, 'float')

-- Define our network
function predict(params, input, target)
   local h1 = pool1(acts1(conv1(params.conv1, reshape(input))))
   local h2 = pool2(acts2(conv2(params.conv2, h1)))
   local h3 = linear(params.linear, flatten(h2))
   local out = util.logSoftMax(h3)
   return out
end

-- Define our loss function
function f(params, input, target)
   local prediction = predict(params, input, target)
   local loss = lossFuns.logMultinomialLoss(prediction, target)
   return loss, prediction
end

-- Get the gradients closure magically:
local df = grad(f, {
   optimize = true,              -- Generate fast code
   stableGradients = true,       -- Keep the gradient tensors stable so we can use CUDA IPC
})

-- Train a neural network
for epoch = 1,100 do
   print('Training Epoch #'..epoch)
   for i = 1,numTrainingBatches() do
      -- Next sample:
      local batch = getTrainingBatch()
      local x = batch.input
      local y = util.oneHot(batch.target, 10)

      -- Grads:
      local grads, loss, prediction = df(params,x,y)

      -- Gather the grads from all nodes
      allReduceSGD.sumAndNormalizeGradients(grads)

      -- Update weights and biases
      for iparam=1,2 do
         params.conv1[iparam] = params.conv1[iparam] - grads.conv1[iparam] * 0.01
         params.conv2[iparam] = params.conv2[iparam] - grads.conv2[iparam] * 0.01
         params.linear[iparam] = params.linear[iparam] - grads.linear[iparam] * 0.01
      end

      -- Log performance:
      confusionMatrix:add(prediction[1], y[1])
      if i % 1000 == 0 then
         -- Reduce confusion matrix so all instances share the same:
         tree.allReduce(confusionMatrix.mat, function(a,b) return a:add(b) end)
         print(confusionMatrix)
         confusionMatrix:zero()
      end
   end

   -- Make sure all nodes are in sync at the end of an epoch
   allReduceSGD.synchronizeParameters(params)
end
