clear all
rng(3); %to have more random wieght initilization , this function specifies the seed parameter of the MATLAB® for random number generator.      
X  = zeros(5, 5, 5);
%white pixel "0", black pixel "1"  
X(:, :, 1) = [ 0 1 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 0 1 0 0;
               0 1 1 1 0
             ];
 
X(:, :, 2) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               1 0 0 0 0;
               1 1 1 1 1
             ];
 
X(:, :, 3) = [ 1 1 1 1 0;
               0 0 0 0 1;
               0 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0
             ];

X(:, :, 4) = [ 0 0 0 1 0;
               0 0 1 1 0;
               0 1 0 1 0;
               1 1 1 1 1;
               0 0 0 1 0
             ];
         
X(:, :, 5) = [ 1 1 1 1 1;
               1 0 0 0 0;
               1 1 1 1 0;
               0 0 0 0 1;
               1 1 1 1 0
             ];
%1-of-N encoding
D = [ 1 0 0 0 0;
      0 1 0 0 0;
      0 0 1 0 0;
      0 0 0 1 0;
      0 0 0 0 1
    ];
%initialize the weights with random real numbers [-1,+1      
W1 = 2*rand(20, 25) - 1;%input, 25 > hidden 1, 20
W2 = 2*rand(20, 20) - 1;%hidden 1, 20 > hidden 2, 20
W3 = 2*rand(20, 20) - 1;%hidden 2, 20 > hidden 3, 20
W4 = 2*rand( 5, 20) - 1;%hidden 3, 20 > output, 5

for epoch = 1:10000           % train for 10,000 times
  [W1, W2, W3, W4] = DeepReLU(W1, W2, W3, W4, X, D);
end

N = 5;                        % inference
for k = 1:N
  x  = reshape(X(:, :, k), 25, 1);
  v1 = W1*x;
  y1 = ReLU(v1);
  
  v2 = W2*y1;
  y2 = ReLU(v2);
  
  v3 = W3*y2;
  y3 = ReLU(v3);
  
  v  = W4*y3;
  y  = Softmax(v)
end
