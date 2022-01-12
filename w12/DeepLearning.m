%W1: input-hidden1, W2: hidden1-hidden2, W3: hidden2-hidden3,
%W4: hidden3-output, X inputs 5x5x5 matrix, D correct outputs
function [W1, W2, W3, W4] = DeepDropout(W1, W2, W3, W4, X, D)
  alpha = 0.0001;%learning rate
  
  N = 5;  %five training data
  for k = 1:N %take the k-th image data 5x5 matrix
    x  = reshape(X(:, :, k), 25, 1);%transform the k-th image
                                   %data 5x5 into 25x1 vector 
                                   
    v1 = W1*x;%weighted sum of the hidden 1
    y1 = ReLU(v1);%output hidden 1 ReLU function
    y1 = y1 .* Dropout(y1, 0.2);
    
    v2 = W2*y1;
    y2 = ReLU(v2);
    y2 = y2 .* Dropout(y2, 0.2);
    
    v3 = W3*y2;
    y3 = ReLU(v3);
    y3 = y3 .* Dropout(y3, 0.2);
   
    v  = W4*y3;%weighted sum of otput
    y  = Softmax(v);%output Softmax Activation function

    d     = D(k, :)';
    e     = d - y;
    delta = e;

    e3     = W4'*delta;
    delta3 = y3.*(1-y3).*e3;
    
    e2     = W3'*delta3;
    delta2 = y2.*(1-y2).*e2;
    
    e1     = W2'*delta2;
    delta1 = y1.*(1-y1).*e1;
    
    dW4 = alpha*delta*y3';
    W4  = W4 + dW4;
    
    dW3 = alpha*delta3*y2';
    W3  = W3 + dW3;
    
    dW2 = alpha*delta2*y1';
    W2  = W2 + dW2;
    
    dW1 = alpha*delta1*x';
    W1  = W1 + dW1;
  end
end
