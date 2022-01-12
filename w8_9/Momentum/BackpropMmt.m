
function [W1, W2] = BackpropMmt(W1, W2, X, D)
  alpha = 0.9; % learning rate
  beta  = 0.9; % momentum factor
  % momentums array size = W weight array size initilized with zerozs
  mmt1 = zeros(size(W1)); 
  mmt2 = zeros(size(W2));
    N = 4;  %Number of training data
  for k = 1:N % Take the K-th of the data points
    x = X(k, :)'; %input viriable (K-th row of X rotated 90^o CW)
    d = D(k); %correct output viriable
    
    v1 = W1*x; %The weighted sum of the hidden nodes
    y1 = Sigmoid(v1);%The output of the hidden nodes    
    v  = W2*y1;%The weighted sum of the output nodes
    y  = Sigmoid(v);%The output from the output nodes
        e     = d - y; %calculate output error
    delta = y.*(1-y).*e; %calculate delta according delta rule

    e1     = W2'*delta;%the error of the hidden nodes
                        %the transpose matrix, W2'
    delta1 = y1.*(1-y1).*e1; %The  back-propagation of the delta
          dW1  = alpha*delta1*x';%the weight update (input->hidden layer)
    mmt1 = dW1 + beta*mmt1;%update momentum value of Layer 1
    W1   = W1 + mmt1; %adjust the weight with momentum (SGD method)
    
    dW2  = alpha*delta*y1';%the weight update (hidden->output layer)  
    mmt2 = dW2 + beta*mmt2;%update momentum value of Layer 2    
    W2   = W2 + mmt2; %adjust the weight with momentum (SGD method)
  end %Repeat the process for N times
end %one epoch done
