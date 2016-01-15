#!/usr/bin/env bash

# run 4 nodes
th mnist.lua --numNodes 4 --nodeIndex 1 &
th mnist.lua --numNodes 4 --nodeIndex 2 --nodePort 8081 &
th mnist.lua --numNodes 4 --nodeIndex 3 --nodePort 8082 &
th mnist.lua --numNodes 4 --nodeIndex 4 --nodePort 8083 &

# wait for them all
wait
