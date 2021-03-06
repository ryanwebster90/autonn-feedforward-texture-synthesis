function G = train_texture_net(varargin)
% Copyright (C) 2017 Ryan Webster
% All rights reserved.
%
% This file is made available under the terms of the MIT license.

opts.N_iter = 50000;
opts.batchsize = 1;
opts.new_figure = true;
opts.N_pool_vgg19 = 3; %No. of pooling layers for vgg19
opts.file_in = 'images/raddish.jpg';
opts.out_dir = './../autonn feedforward results/';

opts = vl_argparse(opts, varargin);

if opts.new_figure
  figure
end
params.weightDecay = 5e-4;
params.learningRate = 5e-2;
params.momentum = 0;
params.solver = [];
params.plotDiagnostics = false;

N_filters = 8;
f_size = 3;


% VGG-19 texture loss net
x = get_normalized_vgg19(opts.N_pool_vgg19);
switch opts.N_pool_vgg19
  case 1
    gm_sz = 128;
  case 2
    gm_sz = 256;
  otherwise
    gm_sz = 512;
end
x = reshape(x,[],gm_sz);
gram_matrix = x'*x/gm_sz^2;

Layer.workspaceNames();
D = Net(gram_matrix);
D.move('gpu');


% generative network
latent = Input();

N_pool = 5;

x = periodic_conv(latent,[],[],'size',[f_size,f_size,N_filters*32,N_filters*16]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters*16,N_filters*8]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters*8,N_filters*4]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters*4,N_filters*2]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters*2,N_filters]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters,3]);
x = vl_nnbnorm(x);
x = vl_nnsigmoid(x);

Layer.workspaceNames() ;
G = Net(x) ;  % compile network
G.move('gpu');


% initialize optimizer parameters
G_state = struct();
G_state.solverState = cell(1, numel(G.params)) ;
G_state.solverState(:) = {0} ;
G_params = params;
G_params.solver = @adam;
G_params.solverOpts = adam();
G_loss_hist = [];


% read input image 
x0 = double(imread(opts.file_in))/255;
x0 = imresize(x0,1,'lanczos3');
x0 = Spectrum.periodic(x0);
x0 = single(gpuArray(x0));

D.eval({'input1',x0},'forward');
x_gram_matrix = D.getValue('gram_matrix');


latent_sz = [size(x0,1)/2^N_pool,size(x0,1)/2^N_pool,N_filters*2^N_pool,opts.batchsize];

out_dir = opts.out_dir;
mkdir(out_dir);
for iter = 1:opts.N_iter
  
  if ~mod(iter,2000);
    params.learningRate = params.learningRate*.7;
  end
  
  latent = randn(latent_sz,'like',x0);
  G.eval({'latent',latent},'forward');
  x_fake = G.getValue('x');
  D.eval({'input1',x_fake},'forward');
  y_gram_matrix = D.getValue('gram_matrix');
  G_der = 2*(y_gram_matrix - repmat(x_gram_matrix,1,1,opts.batchsize));
  G_loss = sum(G_der(:).^2);
  D.eval({'input1',x_fake},'backward',G_der);
  G_der = D.getDer('input1');
  G.eval({'latent',latent},'backward',G_der);
  G_state = accumulateGradientsAutoNN(G, G_state, G_params, 1, []) ;
  
  G_loss_hist = [G_loss_hist,gather(G_loss)];
  
  if ~mod(iter,10)
    subplot(131);
    imshow(x_fake(:,:,:,1))
    drawnow;
    subplot(132);
    imshow(x0)
    drawnow;
    subplot(133);
    hold off;
    plot(G_loss_hist(:))
    drawnow;
  end
  
  if ~mod(iter,1000)
    imwrite(gather(x_fake(:,:,:,1)),[out_dir,num2str(iter),'.jpg']);
  end
end


