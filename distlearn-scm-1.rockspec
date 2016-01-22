package = "distlearn"
version = "scm-1"

source = {
   url = "git://github.com/twitter/torch-distlearn.git",
}

description = {
   summary = "A Distributed Learning library, for Torch",
   homepage = "-",
   license = "MIT",
}

dependencies = {
   "torch >= 7.0",
   "regress",
   "ipc",
}

build = {
   type = "builtin",
   modules = {
      ['distlearn.AllReduceSGD'] = 'lua/AllReduceSGD.lua',
      ['distlearn.AllReduceEA'] = 'lua/AllReduceEA.lua',
   },
}
