--Create training data
require "torch"
dataset={};
function dataset:size() return 100 end -- 100 examples
for i=1,dataset:size() do 
	local input= torch.randn(2);     --normally distributed example in 2d
	local output= torch.Tensor(1);
	if input[1]*input[2]>0 then    --calculate label for XOR function
		output[1]=-1;
	else
		output[1]=1;
	end
	dataset[i] = {input, output};
end


require "nn"
mlp=nn.Sequential();  -- make a multi-layer perceptron


inputs=2; outputs=1; HUs=20;
layer1 = nn.Linear(inputs,HUs);
mlp:add(layer1);
layer2 = nn.Tanh();
mlp:add(layer2);
layer3 = nn.Linear(HUs,outputs);
mlp:add(layer3)

criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)


