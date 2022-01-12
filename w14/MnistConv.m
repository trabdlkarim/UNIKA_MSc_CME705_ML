%W1 conv. filter, W5 pooling-hidden weight, Wo hidden
%-output weight,X 8000x28x28 input training data
%D 8:2(10,000)x1 correct output
function [W1, W5, Wo] = MnistConv(W1, W5, Wo, X, D)
%
alpha = 0.01;
beta  = 0.95;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);% 8000 for training, 2000 for validation

bsize = 100; %bsize selected dataset for mini-batch  
blist = 1:bsize:(N-bsize+1);%location of first training data point
                            % to be brought into the minibatch.
                            %[ 1, 101, 201 .., 7801, 7901 ]
% One epoch loop
for batch = 1:length(blist)% 80 weight updates
  dW1 = zeros(size(W1));
  dW5 = zeros(size(W5));
  dWo = zeros(size(Wo));
  
  % Mini-batch loop
   begin = blist(batch);      %starting point
  for k = begin:begin+bsize-1 % 100 training datasets
    % Forward pass = inference
    %
    x  = X(:, :, k);               % Input,           28x28
    y1 = Conv(x, W1);              % Convolution,  20x20x20
    y2 = ReLU(y1);                 %
    y3 = Pool(y2);                 % Pooling,      10x10x80
    y4 = reshape(y3, [], 1);       %transform the k-th image
                                   %data 10x10x80 into 8000x1 vector
    v5 = W5*y4;                    % ReLU,             2000
    y5 = ReLU(v5);                 %
    v  = Wo*y5;                    % Softmax,          10x1
    y  = Softmax(v);               %

    % One-hot encoding
    %convert numerical correct output into a 10x1 vector
    d = zeros(10, 1);
    d(sub2ind(size(d), D(k), 1)) = 1;

    % Backpropagation
    %
    e      = d - y;                   % Output layer  
    delta  = e;                       %crossentropy function

    e5     = Wo' * delta;             % Hidden(ReLU) layer
    delta5 = (y5 > 0) .* e5;          %derivative of ReLU

    e4     = W5' * delta5;            % Pooling layer
    
    e3     = reshape(e4, size(y3));%transform k-th image 8000x1 vector
                                   % into 10x10x80 data
%pooling and convolution layers
%The explanation of this part is beyond the scope of this book
    e2 = zeros(size(y2));           
    W3 = ones(size(y2)) / (2*2);
    for c = 1:20
      e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
    end
    
    delta2 = (y2 > 0) .* e2;          % ReLU layer
  
    delta1_x = zeros(size(W1));       % Convolutional layer
    for c = 1:20
      delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %sum of wieght updates
    dW1 = dW1 + delta1_x; 
    dW5 = dW5 + delta5*y4';    
    dWo = dWo + delta *y5';
  end % one training dataset done
  
  % Update weights
  %
  dW1 = dW1 / bsize;
  dW5 = dW5 / bsize;
  dWo = dWo / bsize;
  %adjusts the weights using momentum
  momentum1 = alpha*dW1 + beta*momentum1;
  W1        = W1 + momentum1;
  
  momentum5 = alpha*dW5 + beta*momentum5;
  W5        = W5 + momentum5;
   
  momentumo = alpha*dWo + beta*momentumo;
  Wo        = Wo + momentumo;  
end % One weight update done

end% One epoch loop done

