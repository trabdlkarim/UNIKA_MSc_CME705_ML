clear all
              % specify input viriable values
X = [ 0 0 1;  % N=1
      0 1 1;  % N=2
      1 0 1;  % N=3
      1 1 1;  % N=4
    ];
              % Correcet output viriable values
D = [ 0
      1
      1
      0
    ];
%initialize the weights with random real numbers [-1,+1]      
W1 = 2*rand(4, 3) - 1; %input->hidden, 3 inputs,4 hidden
W2 = 2*rand(1, 4) - 1; %hidden->output, 4 hidden, 1 output

for epoch = 1:10000           % train for 10,000 times
                            %call BackpropXOR function
  [W1 W2] = BackpropCE(W1, W2, X, D); 
end

N = 4;                        % inference
for k = 1:N
  x  = X(k, :)';   %K-th row of X rotated 90^o CW
  v1 = W1*x;  %The weighted sum of the hidden nodes      
  y1 = Sigmoid(v1); %The output of the hidden nodes
  v  = W2*y1; %The weighted sum of the output nodes
  y  = Sigmoid(v) %The output from the output nodes
end


