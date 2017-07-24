% initialize net parameters
% field of net:
% sizes: number of neurons in each layer
% nlayers: number of layers
% biases: biases of each layer
% weights: weights of each layer
function [net] = init_net(size)
net.size = size;
net.nlayers = length(size);
% input layer excluded
net.biases = cell(1,net.nlayers - 1);
net.weights = cell(1,net.nlayers - 1);
% initialize using Gaussian distribution
for n = 2:net.nlayers
    net.biases{n - 1} = randn(size(n),1);
    net.weights{n - 1} = randn(size(n),size(n - 1)) / sqrt(size(n - 1));
end
end