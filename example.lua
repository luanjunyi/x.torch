require 'solver'
require 'paths'
require 'nn'
require 'optim'

-- return the sample data(CIFA10), download and normalize each color channel
function prepareCIFA10()
   if (not paths.filep("cifar10torchsmall.zip")) then
      os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
      os.execute('unzip cifar10torchsmall.zip')
   end


   trainset = torch.load('cifar10-train.t7')
   testset = torch.load('cifar10-test.t7')

   local shuffle = torch.randperm(10000):long()

   trainset.data = trainset.data:index(1, shuffle):double()
   trainset.label = trainset.label:index(1, shuffle):double()

   local mean = {}
   local std = {}

   for i = 1,3 do
      mean[i] = trainset.data[{ {}, {i}, {}, {} }]:mean()
      std[i] = trainset.data[{ {},{i},{},{} }]:std()
   end

   for i = 1,3 do
      trainset.data[{ {},{i},{},{} }]:add(-mean[i])
      trainset.data[{ {},{i},{},{} }]:div(std[i])
   end

   XTrain = trainset.data[{ {1,9000},{},{},{} }]
   yTrain = trainset.label[{ {1,9000} }]
   XVal = trainset.data[{ {9001,10000},{},{},{} }]
   yVal = trainset.label[{ {9001,10000} }]


   train = {
      XTrain = XTrain[{ {1, 100},{},{},{} }],
      yTrain = yTrain[{ {1, 100} }],
      XVal = XVal,
      yVal = yVal
   }

   return train
end

-- return a three layer fully connected network
function buildNetwork()
   local model = nn.Sequential()
   model:add(nn.View(3 * 32 * 32))
   model:add(nn.Linear(3 * 32 * 32, 100))
   model:add(nn.Linear(100, 100))
   model:add(nn.Linear(100, 10))
   model:add(nn.LogSoftMax())
   return model
end

-- A minimal example of how to user Sover 
function overFitSmallData(model, train)
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
   return s
end

train = prepareCIFA10()
model = buildNetwork()

s = overFitSmallData(model, train)


-- A more complex example that does random search on hyper parameters
function randomSearchHyperParameters(model, train)
   bestTrainAcc = 0.0
   lr = 0.0006756206653175
   std = 0.074116031115979

   for i = 1, 50 do
      model:reset()
      model = nn.Sequential()
      model:add(nn.View(3 * 32 * 32))
      model:add(nn.Linear(3 * 32 * 32, 100))
      model:add(nn.Linear(100, 100))
      model:add(nn.Linear(100, 10))
      model:add(nn.LogSoftMax())

      local learningRate = torch.pow(10, -torch.uniform(1, 5))
      local weightStd = torch.pow(10, -torch.uniform(1, 5))

      print('run ' .. i .. ': learning rate: ' .. learningRate .. ' weigth std: ' .. weightStd)

      s = Solver(model, train,
                 {
                    criterion = nn.ClassNLLCriterion(),
                    optimize = optim.adam,
                    optimConfig = {learningRate = learningRate},
                    initWeight = true,
                    weightStd = weightStd,
                    numEpoches = 20,
                    verbose = true,
                 }
      )

      s:train()
      local cur = torch.Tensor(s.trainAccHistory)
      print(#s.trainAccHistory)
      print(cur[{ {-5, -1} }])
      local cur = cur[{ {-5, -1} }]:mean()
      print('current accuracy: ' .. cur)
      if cur > bestTrainAcc then
         bestTrainAcc = cur
         lr = learningRate
         std = weightStd
         print('======= better acc: ' .. cur)
      end


      print(model:parameters()[1]:mean())
   end
end
