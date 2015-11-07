--Create training data
require "torch"

function tablelength(T)
  local count = 0
  for _ in pairs(T) do count = count + 1 end
  return count
end


function to_dataset(data_x, data_y)
  local dataset = {};
  local data_dim = tablelength(data_x);

  local data_size = tablelength(data_y["Sales"]);
  for i = 1, data_size do
    local output = torch.Tensor(1); 
    local input = torch.Tensor(data_dim);
    output[1] = data_y["Sales"][i]/7000;
    local j = 1
    for k, v in pairs(data_x) do
      if j == 14 then
         input[j] = 0;
      else
          input[j] = v[i];
      end
      j = j + 1
    end
    dataset[i] = {input, output}
  end
  return dataset
end



function eval_error(data, mlp)
  s = 0;
  for i = 1, #data do
    y = data[i][2][1]*7000;
    if y > 0 then
      o = (mlp:forward(data[i][1])*7000)[1];
      
      r2 = ((o-y)/y)^2;
      s = s + r2;
    end
  end
  return torch.sqrt(s/#data);
end

require "csvigo"
train_set = to_dataset(csvigo.load("store120_train_x.csv"), csvigo.load("store120_train_y.csv"))
function train_set:size() return #train_set end

valid_set = to_dataset(csvigo.load("store120_valid_x.csv"), csvigo.load("store120_valid_y.csv"))
function valid_set:size() return #valid_set end

valid2_set = to_dataset(csvigo.load("store120_valid2_x.csv"), csvigo.load("store120_valid2_y.csv"))
function valid2_set:size() return #valid2_set end


require "nn"
mlp=nn.Sequential();  -- make a multi-layer perceptron


inputs= train_set[1][1]:size()[1]; outputs=1; HUs=4;
layer1 = nn.Linear(inputs,HUs);
mlp:add(layer1);
layer2 = nn.Tanh();
mlp:add(layer2);
layer3 = nn.Linear(HUs,outputs);
mlp:add(layer3)

criterion = nn.MSECriterion()  
trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.001
trainer:train(train_set)


