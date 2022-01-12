%enter the second image (k = 2) of the test data into the
%neural network and display the results of all the steps.
clear all

load('MnistConv.mat')

k  = 2;
x  = X(:, :, k);
y1 = Conv(x, W1);                 % Convolution,  20x20x20
y2 = ReLU(y1);                    %
y3 = Pool(y2);                    % Pool,         10x10x20
y4 = reshape(y3, [], 1);          %                   2000  
v5 = W5*y4;                       % ReLU,              360
y5 = ReLU(v5);                    %
v  = Wo*y5;                       % Softmax,            10
y  = Softmax(v);                  %
  

figure;
display_network(x(:));            %display 28x28 input image of
title('Input Image')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
convFilters = zeros(9*9, 20);
for i = 1:20
  filter            = W1(:, :, i);
  convFilters(:, i) = filter(:);
end

figure
display_network(convFilters);%display (9x9) 20 trained convolution filters
                         %The greater the value, the brighter the shade
                         %the best features extracted from the MNIST image
title('Convolution Filters')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fList = zeros(20*20, 20);
for i = 1:20
  feature     = y1(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);%(20x20)x20 results (y1) of image
                       %processing of conv. layer
title('Features [Convolution]')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fList = zeros(20*20, 20);
for i = 1:20
  feature     = y2(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);% (20x20)x20 ReLU function processed on feature map
% The dark pixels of the previous image are removed, and the current
%images have mostly white pixels on the letter
title('Features [Convolution + ReLU]')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fList = zeros(10*10, 20);
for i = 1:20
  feature     = y3(:, :, i);
  fList(:, i) = feature(:);
end
figure
display_network(fList);%(10x10)x20 images after the mean pooling process
%half the previous size ==> pooling layer can reduce the required resources
title('Features [Convolution + ReLU + MeanPool]')