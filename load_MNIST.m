function [training_set,validation_set,test_set] = load_MNIST()
% separate 10000 of training set as validation set
[training_set,validation_set] = loadset('MNIST/train-labels.idx1-ubyte','MNIST/train-images.idx3-ubyte',10000);

test_set = loadset('MNIST/t10k-labels.idx1-ubyte','MNIST/t10k-images.idx3-ubyte',0);
end

function [set,validation_set] = loadset(labelfile,imagefile,nvalidations)
labelfid = fopen(labelfile,'r');
% 2049 is the magic number for label files
if fread(labelfid,1,'int32','b') ~= 2049
    error 'Invalid label file£¡Wrong maigic number!'
end
imagefid = fopen(imagefile,'r');
% 2051 is the magic number for label files
if fread(imagefid,1,'int32','b') ~= 2051
    error 'Invalid image file£¡Wrong maigic number!'
end

% nlabels and nimages are the numbers of labels and images
nlabels = fread(labelfid,1,'int32','b');
nimages = fread(imagefid,1,'int32','b');
if(nlabels ~= nimages)
    error 'Labels and images do not match!'
end
fprintf('%d samples will be loaded\n',nlabels)

% nrows and ncols is the numbers of image's height and width
nrows = fread(imagefid,1,'int32','b');
ncols = fread(imagefid,1,'int32','b');
fprintf('Each image is %d*%d\n',nrows,ncols)

% read labels
[labels,cnt] = fread(labelfid);
if cnt ~= nlabels
    error 'Invalid label file£¡Not enough labels!'
end
labels = num2cell(labels);
% not used by net
[set(1:nlabels - nvalidations).label] = labels{1:nlabels - nvalidations};
[validation_set(1:nvalidations).label] = labels{nlabels - nvalidations+1:end};
% vectorize labels
[labels] = cellfun(@vectorize_label,labels,'UniformOutput',false);
[set(1:nlabels - nvalidations).y] = labels{1:nlabels - nvalidations};
[validation_set(1:nvalidations).y] = labels{nlabels - nvalidations+1:end};

% read images
for n = 1:nimages-nvalidations
    [image, cnt] = fread(imagefid,[ncols nrows]);
    % fread reads data in column order, transpose needed
    % and normalize pixel value
    set(n).image = image' / 255; % not used by net
    set(n).x = reshape(set(n).image,[],1);

    if cnt ~= nrows * ncols
        error 'Invalid image file£¡Not enough pixels!'
    end
end
for n = 1:nvalidations
    [image, cnt] = fread(imagefid,[ncols nrows]);
    % fread reads data in column order, transpose needed
    % and normalize pixel value
    validation_set(n).image = image' / 255; % not used by net
    validation_set(n).x = reshape(validation_set(n).image,[],1);

    if cnt ~= nrows * ncols
        error 'Invalid image file£¡Not enough pixels!'
    end
end

fclose(labelfid);
fclose(imagefid);
end

% change label to vector
% e.g. 5 to 0;0;0;0;0;1;0;0;0;0
function vec = vectorize_label(label)
vec =  zeros(10,1);
vec(label + 1) = 1;
end