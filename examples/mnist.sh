#!/usr/bin/env bash

# run 4 nodes
th mnist.lua --numNodes 4 --nodeIndex 1 &
th mnist.lua --numNodes 4 --nodeIndex 2 &
th mnist.lua --numNodes 4 --nodeIndex 3 &
th mnist.lua --numNodes 4 --nodeIndex 4 &

# wait for them all
wait
