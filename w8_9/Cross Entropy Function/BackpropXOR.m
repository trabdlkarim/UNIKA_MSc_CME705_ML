
%W1 weight matrix between the input layer and hidden layer
%W2 weight matrix between the hidden layer and output layer
%X inputs, D correct outputs
function [W1, W2] = BackpropXOR(W1, W2, X, D) 
  alpha = 0.9; % learning rate
  
  N = 4;       %Number of training data
  for k = 1:N  % Take the K-th of the data points
    x = X(k, :)';  %input viriable (K-th row of X rotated 90^o CW)
    d = D(k);      %correct output viriable
    
    v1 = W1*x;        %The weighted sum of the hidden nodes
    y1 = Sigmoid(v1); %The output of the hidden nodes   
    v  = W2*y1;       %The weighted sum of the output nodes
    y  = Sigmoid(v);  %The output from the output nodes
    
    e     = d - y;    %calculate output error
    delta = y.*(1-y).*e; %calculate delta according delta rule

    e1     = W2'*delta; %the error of the hidden nodes
                        %the transpose matrix, W2'
    delta1 = y1.*(1-y1).*e1; %The  back-propagation of the delta
                             %of the hidden nodes using delta rule
    %the element-wise product operator, .*, is used because the 
    %variables are vectors. It performs an operation on each
    %element of the vector
    dW1 = alpha*delta1*x'; %the weight update (input->hidden layer)
    W1  = W1 + dW1; %Immediately adjust the weight (SGD method)
    
    dW2 = alpha*delta*y1'; %the weight update (hidden->output layer)  
    W2  = W2 + dW2; %Immediately adjust the weight (SGD method)
  end %Repeat the process for N times
end %one epoch done

