%W1 weight matrix for the input-hidden layer
%W2 weight matrix for the hidden-output layer
%X inputs 5x5x5 matrix, D correct outputs
function [W1, W2] = MultiClass(W1, W2, X, D)
  alpha = 0.9;
  
  N = 5; %five training data 
  for k = 1:N %take the k-th image data 5x5 matrix
    x = reshape(X(:, :, k), 25, 1); %transform the k-th image
                                   %data 5x5 into 25x1 vector
    d = D(k, :)';
    
    v1 = W1*x;  %The weighted sum of the hidden nodes
    y1 = Sigmoid(v1);%Activation function of the hidden nodes
    v  = W2*y1; %The weighted sum of the output nodes
    y  = Softmax(v);%Activation function of the output nodes
    
    e     = d - y;
    delta = e;%learning rule of the cross entropy function

    e1     = W2'*delta; %the error of the hidden nodes
                        %the transpose matrix, W2'
    delta1 = y1.*(1-y1).*e1;% back-propagated delta
                             %of the hidden nodes  
    
    dW1 = alpha*delta1*x'; %the weight update (input->hidden layer)
    W1  = W1 + dW1; %Immediately adjust the weight (SGD method)
    
    dW2 = alpha*delta*y1'; %the weight update (hidden->output layer)  
    W2  = W2 + dW2; %Immediately adjust the weight (SGD method)
  end %Repeat the process for N times
end %one epoch done
