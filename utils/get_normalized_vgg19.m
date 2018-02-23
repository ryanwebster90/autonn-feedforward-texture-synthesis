function x = get_normalized_vgg19(N_pool)
% returns the final layer of the VGG19 convolutional net
% specified as the first relu after N_pool
%
% Copyright (C) Ryan Webster, 2018



tmp = load(['./../models/texture-synth-weights/vgg_normalised_conv',num2str(N_pool+1),'_1.mat']);
x = Input();

% ENCODER

% normalize input
w = single(gpuArray(tmp.conv1_weight));
w = permute(w,[3,4,2,1]);
x = vl_nnconv(x,w,single(gpuArray(tmp.conv1_bias(:))));

i = 2;
for pool = 1:N_pool
  
  if pool == 1
    N_blocks = 2;
  elseif pool >=3
    N_blocks = 3;
  else
    N_blocks = 1;
  end
  
  for block = 1:N_blocks
    w = single(gpuArray(tmp.(['conv',num2str(i),'_weight'])));
    w = permute(w,[3,4,2,1]);
    b = single(gpuArray(tmp.(['conv',num2str(i),'_bias'])));
    x = periodic_conv(x,w,b);
    x = vl_nnrelu(x);
    i = i+1;
  end
  
  x = vl_nnpool(x,[2 2],'method','avg','stride', [2 2],'pad',[0 1 0 1]);
  
  w = single(gpuArray(tmp.(['conv',num2str(i),'_weight'])));
  w = permute(w,[3,4,2,1]);
  b = single(gpuArray(tmp.(['conv',num2str(i),'_bias'])));
  x = periodic_conv(x,w,b);
  x = vl_nnrelu(x);
  i = i+1;
  
end

x.name = 'output';
