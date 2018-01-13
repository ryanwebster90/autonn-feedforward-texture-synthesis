function [G,D] = train_texture_net(varargin)
% Custom matlab implementation of "Texture Networks" by Ulyanov et al
%
% Copyright (C) 2018 Ryan Webster
% All rights reserved.
%
% This file is made available under the terms of the MIT license.

opts.N_iter = 2000;
opts.batchsize = 1;
opts.clip = 1;
opts.new_figure = true;
opts.N_channels = 3;
% opts.patchsize = 48;
opts = vl_argparse(opts, varargin);

if opts.new_figure
  
  figure
end
params.weightDecay = 5e-4;
params.learningRate = 5e-3;
params.momentum = 0;
params.solver = [];
params.plotDiagnostics = false;

N_filters = 32;
f_size = 5;
ps = 2;

% discriminator
D = load('vgg19_relu3_1.mat');
D = Net(D);
layers = Layer.fromCompiledNet(D);
x = layers{1};
x = reshape(x,[],256,opts.batchsize);
gram_matrix = x.'*x;
Layer.workspaceNames();
D = Net(gram_matrix);
D.move('gpu');


%generative net
latent = Input();

N_pool = 1;
% x = vl_nnbnorm(latent);
x = vl_nnconv(latent,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters*16,N_filters*8]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

% x = vl_nnconv(x,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters*8,N_filters*4]) ;
% x = vl_nnbnorm(x);
% x = vl_nnrelu(x);
% x = repelem(x,2,2);
% 
% x = vl_nnconv(x,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters*4,N_filters*2]) ;
% x = vl_nnbnorm(x);
% x = vl_nnrelu(x);
% x = repelem(x,2,2);
% 
% x = vl_nnconv(x,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters*2,N_filters]) ;
% x = vl_nnbnorm(x);
% x = vl_nnrelu(x);
% x = repelem(x,2,2);

x = vl_nnconv(x,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters,opts.N_channels]) ;
x = vl_nnbnorm(x);
x = vl_nnsigmoid(x);
% x = tanh_layer(x);

Layer.workspaceNames() ;
G = Net(x) ;  % compile network
G.move('gpu');

G_state = struct();
G_state.solverState = cell(1, numel(G.params)) ;
G_state.solverState(:) = {0} ;
G_params = params;
G_params.solver = @adam;
G_params.solverOpts = adam();

G_loss_hist = [];


x0 = double(imread('campbell.jpg'))/255;
x0 = imresize(x0,1,'lanczos3');
x0 = single(gpuArray(x0));
% x0 = 2*x0-1;
if opts.N_channels == 1;
  x0 = mean(x0,3);
end

D.eval({'input',x0},'forward');
x_gram_matrix = D.getValue('gram_matrix');


latent_sz = [size(x0,1)/2^N_pool,size(x0,1)/2^N_pool,N_filters*2^N_pool,opts.batchsize];

for iter = 1:opts.N_iter
  if ~mod(iter,1000);
    params.learningRate = min(params.learningRate*.75,5e-5);
  end
  
  
  latent = randn(latent_sz,'like',x0);
  G.eval({'latent',latent},'forward');
  x_fake = G.getValue('x');
  D.eval({'input',x_fake},'forward');
  y_gram_matrix = D.getValue('gram_matrix');
  G_der = 2*(y_gram_matrix - x_gram_matrix);
  G_loss = sum(G_der(:).^2);
  D.eval({'input',x_fake},'backward',G_der);
  G_der = D.getDer('input');
  G.eval({'latent',latent},'backward',G_der);
  G_state = accumulateGradientsAutoNN(G, G_state, G_params, opts.batchsize, []) ;
  
  G_loss_hist = [G_loss_hist,gather(G_loss)];
  
  if ~mod(iter,10)
    subplot(131);
    imshow(x_fake)
    drawnow;
    subplot(132);
    imshow(x0)
    drawnow;
    subplot(133);
    hold off;
    plot(G_loss_hist(:))
    drawnow;
  end
  
end


latent = randn([latent_sz,9],'like',x0);
G.eval({'latent',latent},'forward');
x_fake = G.getValue('x');
figure
display_tensor(x_fake,3);



