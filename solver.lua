require 'nn'
require 'optim'

local Solver = torch.class('Solver')

function Solver:__init(model,
                       data,
                       configs)
   self.model = model
   self.data = data
   local c = configs
   self.criterion = c.criterion or nn.ClassNLLCriterion()
   self.optimize = c.optimize or optim.adam
   self.optimConfig = c.optimConfig or {}
   self.lrDecay = c.lrDecay or 1.0
   self.batchSize = c.batchSize or 20
   self.numEpoches = c.numEpoches or 40
   self.printEvery = c.printEvery or 100
   self.verbose = c.verbose or false
   self.initWeight = c.initWeight or false
   self.weightStd = c.weightStd

   self.XTrain = data.XTrain
   self.yTrain = data.yTrain
   self.XVal = data.XVal
   self.yVal = data.yVal


   self.firstTrain = true
   self:reset()

   print('created new solver')
end

function Solver:reset()
   self.epoch = 0
   self.bestValAcc = 0
   self.bestParams = nil
   self.lossHistory = {}
   self.trainAccHistory = {}
   self.valAccHistory = {}
   self.optimState = {}
end

function Solver:checkAccuracy(X, y, sampleSize)
   local sampleSize = sampleSize or y:size()[1]
   if sampleSize > y:size()[1] then
      sampleSize = y:size()[1]
   end

   local shuffle = torch.randperm(y:size()[1])[{ {1, sampleSize} }]:long()
   X = X:index(1, shuffle)
   y = y:index(1, shuffle)
   
   local output = self.model:forward(X)
   local _;
   local yPredict;
   _, yPredict = torch.sort(output, true)

   yPredict = yPredict[{ {},{1} }]:reshape(y:size()):long()
   y = y:long()
   local acc = y:eq(yPredict):sum() / sampleSize
   return acc
end

function Solver:step(XBatch, yBatch)
   local parameters, grads = self.model:getParameters()

   local feval = function(x)
      -- parameters:copy(x)
      grads:zero()
      local output = self.model:forward(XBatch)
      local loss = self.criterion:forward(output, yBatch) / yBatch:size()[1]
      local delta = self.criterion:backward(output, yBatch)
      self.model:backward(XBatch, delta)

      table.insert(self.lossHistory, loss)

      -- if self.verbose then
      --    print('loss: ' .. loss)
      -- end

      return loss, grads
   end
   
   self.optimize(feval, parameters, self.optimConfig, self.optimState)
end

function Solver:train()

   if self.initWeight and self.firstTrain then
      -- initWeights will change the underlying storage of weights
      -- if there are existing weightGrads, later call will check if the
      -- weights and weighGrads have same storageOffset. So if it's not
      -- first run, we should skip initWeight. It doesn't make sense to 
      -- init the weights if the solver has been trained anyway.
      self:initWeights(self.weightStd)
   end

   self.model:training()

   numTrain = self.XTrain:size()[1]
   for t = 1, self.numEpoches do
      shuffle = torch.randperm(numTrain):long()

      print('epoch ' .. t .. ": " .. self.model:parameters()[1]:mean())
      for i = 1, numTrain, self.batchSize do
         shuffledIndex = shuffle[{ {i, math.min(i + self.batchSize - 1, numTrain)} }]
         XBatch = self.XTrain:index(1, shuffledIndex)
         yBatch = self.yTrain:index(1, shuffledIndex)
         self:step(XBatch, yBatch)
      end

      local trainAcc = self:checkAccuracy(self.XTrain, self.yTrain, 1000)
      local valAcc = self:checkAccuracy(self.XVal, self.yVal)
      table.insert(self.trainAccHistory, trainAcc)
      table.insert(self.valAccHistory, valAcc)
      if valAcc > self.bestValAcc then
         self.bestValAcc = valAcc
         self.bestParams, _ = self.model:parameters()
         self.bestParams = self:cloneParams(self.bestParams)
      end
      if self.verbose then
         print(trainAcc .. " -  " .. valAcc)
      end
   end

   self.firstTrain = false
end

function Solver:cloneParams(params)
   ret = {}
   for k, v in ipairs(params) do
      ret[k] = v:clone()
   end
   return ret
end

function Solver:initWeights(std)
   print('init weight with std = ' .. std .. ' learning rate = ' .. self.optimConfig.learningRate)
   for i = 1, #self.model do
      layer = self.model:get(i)
      if layer.weight then
         layer.weight = torch.randn(layer.weight:size())
         layer.weight:mul(std)
      end
   end
end
