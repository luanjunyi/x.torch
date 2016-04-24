# x.torch
Torch 7 is intended to be Theano+Keras. But it's not there. For example, Torch 7 has the awesome `add` methods in the containers similar to other higher level lib. However, there's no explicite support of vanilla SGD. You've to write your own training process.

This repo contains modules that should be in Torch 7 but not which are necessary to bring Torch 7 to Keras level. The first release contains a Solver class inspired by Caffe and CS231n class offered by Stanford. I'll gradually add more functions.

# Sample code using Solver

Three parts need to initialize a solver that encapusulates a vanilla SGD.

1. Data: a table contains 4 keys: XTrain, yTrain, XVal, yVal. XVal and yVal are validation set
2. Model: a `container` that contains many layers
3. Various optimization configurations

Below is a snippet from the example.lua file. See that file for more detail.

```lua
-- data is training data contains the 4 keys
-- model is a network

local lr = 0.0006756206653175
local std = 0.074116031115979

local s = Solver(model, train,
  {
     criterion = nn.ClassNLLCriterion(),
     optimize = optim.adam,
     optimConfig = {learningRate = lr},
     initWeight = true,
     weightStd = std,
     numEpoches = 20,
     verbose = true,
  }
)
s:train()
```

`lr` and `std` should be picked very carefully otherwise the training will not work well
