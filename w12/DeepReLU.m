%W1: input-hidden1, W2: hidden1-hidden2, W3: hidden2-hidden3,
%W4: hidden3-output, X inputs 5x5x5 matrix, D correct outputs
function [W1, W2, W3, W4] = DeepReLU(W1, W2, W3, W4, X, D)
  alpha = 0.01;%learning rate
  
  N = 5;  %five training data
  for k = 1:N %take the k-th image data 5x5 matrix
    x  = reshape(X(:, :, k), 25, 1);%transform the k-th image
                                   %data 5x5 into 25x1 vector 
    d     = D(k, :)';
    
    v1 = W1*x;%weighted sum of the hidden 1
    y1 = ReLU(v1);%output hidden 1 ReLU Activation function
                  
    v2 = W2*y1;%weighted sum of the hidden 2
    y2 = ReLU(v2);%output hidden 2 ReLU Activation function
    
    v3 = W3*y2;%weighted sum of the hidden 3
    y3 = ReLU(v3);%output hidden 3 ReLU Activation function
    
    v  = W4*y3;%weighted sum of otput
    y  = Softmax(v);%output Softmax Activation function
 
    e     = d - y;%output error 
    delta = e;%learning rule of the cross entropy function

    e3     = W4'*delta; %hidden nodes 3 error
    delta3 = (v3 > 0).*e3;% back-propagated delta 3
                          %derivative of ReLU
    e2     = W3'*delta3;
    delta2 = (v2 > 0).*e2;% back-propagated delta 2
                          %derivative of ReLU
    e1     = W2'*delta2;
    delta1 = (v1 > 0).*e1;% back-propagated delta 1
                          %derivative of ReLU
    dW4 = alpha*delta*y3';
    W4  = W4 + dW4; %the weight update
    
    dW3 = alpha*delta3*y2';
    W3  = W3 + dW3; %the weight update
    
    dW2 = alpha*delta2*y1';
    W2  = W2 + dW2; %the weight update
    
    dW1 = alpha*delta1*x';
    W1  = W1 + dW1; %the weight update
  end
end
