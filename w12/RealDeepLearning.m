clear all
 
TestDeepDropout;%train to load W1, W2, W3, W4
X  = zeros(5, 5, 5);
 
X(:, :, 1) = [ 0 0 1 1 0;
               0 0 1 1 0;
               0 1 0 1 0;
               0 0 0 1 0;
               0 1 1 1 0
             ];
 
X(:, :, 2) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 1;
               1 1 1 1 1
             ];
 
X(:, :, 3) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 1;
               1 1 1 1 0
             ];
         
X(:, :, 4) = [ 0 1 1 1 0;
               0 1 0 0 0;
               0 1 1 1 0;
               0 0 0 1 0;
               0 1 1 1 0
             ];     
         
X(:, :, 5) = [ 0 1 1 1 1;
               0 1 0 0 0;
               0 1 1 1 0;
               0 0 0 1 0;
               1 1 1 1 0
             ];    
         
N = 5;                        % inference
for k = 1:N
  x  = reshape(X(:, :, k), 25, 1);
  v1 = W1*x;
  y1 = Sigmoid(v1);
  
  v2 = W2*y1;
  y2 = Sigmoid(v2);
  
  v3 = W3*y2;
  y3 = Sigmoid(v3);
  
  v  = W4*y3;
  y  = Softmax(v)
end
