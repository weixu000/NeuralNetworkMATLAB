% example code to run
[trs,~,tes] = load_MNIST();
net = init_net([28*28 40 20 10]);
net = SGD(net,50,50,0.1,0.05,trs,tes);