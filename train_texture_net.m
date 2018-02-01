function G = train_texture_net(varargin)
% Copyright (C) 2018 Ryan Webster
% All rights reserved.
%
% This file is made available under the terms of the MIT license.

opts.N_iter = 50000;
opts.clip = 1;
opts.new_figure = true;
opts.N_channels = 3;
% opts.patchsize = 48;
opts = vl_argparse(opts, varargin);

if opts.new_figure
  figure
end
params.weightDecay = 5e-4;
params.learningRate = 1e-2;
params.momentum = 0;
params.solver = [];
params.plotDiagnostics = false;

N_filters = 16;
f_size = 3;
ps = 2;

% discriminator
x = periodic_vgg19();
x = reshape(x,[],256);
gram_matrix = x'*x;
Layer.workspaceNames();
D = Net(gram_matrix);
D.move('gpu');


%generative net
latent = Input();

N_pool = 4;

x = periodic_conv(latent,[],[],'size',[f_size,f_size,N_filters*16,N_filters*8]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = cat(3,x,randn(size(x),'single'));
x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters*8*2,N_filters*4]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = cat(3,x,randn(size(x),'single'));
x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters*4*2,N_filters*2]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = cat(3,x,randn(size(x),'single'));
x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters*2*2,N_filters]);
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = cat(3,x,randn(size(x),'single'));
x = periodic_conv(x,[],[],'size',[f_size,f_size,N_filters*2,3]);
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


x0 = double(imread('raddish.jpg'))/255;
x0 = imresize(x0,1,'lanczos3');
x0 = Spectrum.periodic(x0);
x0 = single(gpuArray(x0));
% x0 = 2*x0-1;
if opts.N_channels == 1;
  x0 = mean(x0,3);
end

D.eval({'input1',x0},'forward');
x_gram_matrix = D.getValue('gram_matrix');


latent_sz = [size(x0,1)/2^N_pool,size(x0,1)/2^N_pool,N_filters*2^N_pool,1];

out_dir = './tnet results/tnet scale noise sc.5 lr1e-1/';
mkdir(out_dir);

for iter = 1:opts.N_iter
  
  if ~mod(iter,2500);
    params.learningRate = params.learningRate*.7;
    params.learningRate = max(params.learningRate,1e-4);
    imwrite(gather(x_fake),[out_dir,num2str(iter),'.jpg']);
  end
  
  latent = randn(latent_sz,'like',x0);
  G.eval({'latent',latent},'forward');
  x_fake = G.getValue('x');
  D.eval({'input1',x_fake},'forward');
  y_gram_matrix = D.getValue('gram_matrix');
  G_der = 2*(y_gram_matrix - x_gram_matrix);
  G_loss = sum(G_der(:).^2);
  D.eval({'input1',x_fake},'backward',G_der);
  G_der = D.getDer('input1');
  G.eval({'latent',latent},'backward',G_der);
  G_state = accumulateGradientsAutoNN(G, G_state, G_params, 1, []) ;
  
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


