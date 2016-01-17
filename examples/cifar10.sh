#!/usr/bin/env bash

# run 4 nodes
th cifar10.lua --numNodes 4 --nodeIndex 1 &
th cifar10.lua --numNodes 4 --nodeIndex 2 &
th cifar10.lua --numNodes 4 --nodeIndex 3 &
th cifar10.lua --numNodes 4 --nodeIndex 4 &

# wait for them all
wait
