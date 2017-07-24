% stochastic gradient descent
% nepochs: number of epochs trained for
% eta: learning rate
% lambda: regularization parameter
function net = SGD(net,nepochs,nminibat,eta,lambda,training_set,test_set)
ntraining = length(training_set); % number of trainning samples

for epoch = 1:nepochs
    % shuffle the training set
    training_set = training_set(randperm(ntraining)); 
    % divide the training set into minibatchs and update the network
    for batch = 1:nminibat:ntraining
        net = update_minibat(net,eta,lambda,ntraining,training_set(batch:min([batch + nminibat - 1,ntraining])));
    end
    
    fprintf('Epoch %d accuracy:%.2f%%\n',epoch,eval_accur(net,test_set) * 100)
end
end

function net = update_minibat(net,eta,lambda,ntraining,minibat)
    nminibat = length(minibat); % number of trainning samples
    % xs,ys: inputs/outputs of the net
    [xs{1:nminibat}] = minibat(:).x;
    xs = cell2mat(xs);
    [ys{1:nminibat}] = minibat(:).y;
    ys = cell2mat(ys);
    % as,zs: activitions/weighted inputs of the net
    as = cell(1,net.nlayers);
    zs = cell(1,net.nlayers - 1);
    as{1} = xs;
    % feedforward to fill as,zs
    for n = 1:net.nlayers - 1
        zs{n} = net.weights{n} * as{n} + repmat(net.biases{n},1,nminibat);
        as{n + 1} = sigmoid(zs{n});
    end

    % backpropagccation
    % nabla_b,nabla_w: sum of gradients of all the inputs
    nabla_b = cell(1,net.nlayers - 1);
    nabla_w = cell(1,net.nlayers - 1);
    % deltas: delta for the layer,1 column for 1 input
    delta = as{net.nlayers} - ys;
    nabla_b{net.nlayers - 1} = sum(delta,2);
    nabla_w{net.nlayers - 1} = delta * as{net.nlayers - 1}';
    for n = net.nlayers - 2:-1:1
        delta = (net.weights{n + 1}' * delta) .* sigmoid_prime(zs{n});
        nabla_b{n} = sum(delta,2);
        nabla_w{n} = delta * as{n}';
    end

    % gradient descent
    update_weight_fun = @(w,nabla) (1 - eta * (lambda / ntraining)) * w - (eta / nminibat) * nabla;
    update_bias_fun = @(b,nabla) b - (eta / nminibat) * nabla;
    net.weights = cellfun(update_weight_fun,net.weights,nabla_w,'UniformOutput',false);
    net.biases = cellfun(update_bias_fun,net.biases,nabla_b,'UniformOutput',false);
end

% calculate accuracy rate over the set
function accuracy = eval_accur(net,set)
    accuracy = 0;
    for m = 1:length(set)
        a = set(m).x;
        for n = 1: net.nlayers - 1
            a = sigmoid(net.weights{n} * a + net.biases{n});
        end
        [~,ouput] = max(a);
        [~,label] = max(set(m).y);
        if ouput == label
            accuracy = accuracy + 1;
        end
    end
    accuracy = accuracy / length(set);
end

function a = sigmoid(z)
a = 1 ./ (1 + exp(-z));
end

% derivative of sigmoid
function prime = sigmoid_prime(z)
prime = sigmoid(z) .* (1-sigmoid(z));
end