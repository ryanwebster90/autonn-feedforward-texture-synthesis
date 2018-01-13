function [G,D] = train_texture_spatial_gan(varargin)
% Custom matlab implementation of "Spatial Adversarial Networks" by Zalando
% Research
% 
% Copyright (C) 2018 Ryan Webster
% All rights reserved.
%
% This file is made available under the terms of the MIT license.

opts.N_iter = 10000; 
opts.batchsize = 64;
opts.clip = 1;
opts.new_figure = true;
opts.N_channels = 3;
opts.patchsize = 64;
opts = vl_argparse(opts, varargin);

if opts.new_figure
  
figure
end
params.weightDecay = 5e-4;
params.learningRate = 4e-4;
params.momentum = 0;
params.solver = [];
params.plotDiagnostics = false;

N_filters = 64;
f_size = 5;
ps = 2;
N_pool = 4;

% discriminator
images = Input();
x = vl_nnconv(images, 'size', [f_size,f_size, opts.N_channels, N_filters],'pad',ps*[1 1 1 1]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = vl_nnpool(x, 2, 'stride', 2,'method','avg') ;
% x = images;
x = vl_nnconv(x, 'size', [f_size,f_size, N_filters, N_filters*2],'pad',ps*[1 1 1 1]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = vl_nnpool(x, 2, 'stride', 2,'method','avg') ;

x = vl_nnconv(x, 'size', [f_size,f_size, N_filters*2, N_filters*4],'pad',ps*[1 1 1 1]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = vl_nnpool(x, 2, 'stride', 2,'method','avg') ;

x = vl_nnconv(x, 'size', [f_size,f_size, N_filters*4, N_filters*8],'pad',ps*[1 1 1 1]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = vl_nnpool(x, 2, 'stride', 2,'method','avg') ;
%output = 4x4

x = vl_nnconv(x, 'size', [f_size,f_size, N_filters*8, 1],'pad',ps*[1 1 1 1]) ;
% x = vl_nnbnorm(x);
% pred = vl_nnsigmoid(x);
pred = x;
Layer.workspaceNames();

D = Net(pred) ;
D.move('gpu');


%generative net
latent = Input();
x = vl_nnconv(latent,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters*16,N_filters*8]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);
% x = latent;
x = vl_nnconv(x,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters*8,N_filters*4]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = vl_nnconv(x,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters*4,N_filters*2]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = vl_nnconv(x,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters*2,N_filters]) ;
x = vl_nnbnorm(x);
x = vl_nnrelu(x);
x = repelem(x,2,2);

x = vl_nnconv(x,'pad',ps*[1 1 1 1], 'size', [f_size,f_size,N_filters,opts.N_channels]) ;
x = vl_nnbnorm(x);
x = vl_nnsigmoid(x);
% x = tanh_layer(x);

Layer.workspaceNames() ;
G = Net(x) ;  % compile network
G.move('gpu');

D_state = struct();
D_state.solverState = cell(1, numel(D.params)) ;
D_state.solverState(:) = {0} ;
D_params = params;
D_params.solver = @adam;
D_params.solverOpts = adam();

G_state = struct();
G_state.solverState = cell(1, numel(G.params)) ;
G_state.solverState(:) = {0} ;
G_params = params;
G_params.solver = @adam;
G_params.solverOpts = adam();



G_loss_hist = []; D_loss_hist = [];

fn = 'raddish.jpg';
x0 = double(imread(fn))/255;
x0 = Spectrum.periodic(x0);
x0 = imresize(x0,1,'lanczos3');
x0 = single(gpuArray(x0));
% x0 = 2*x0-1;
if opts.N_channels == 1;
  x0 = mean(x0,3);
end

write_dir = ['./results/',fn,' 2/'];
mkdir(write_dir);

dataratio = min(opts.batchsize/(size(x0,1)*size(x0,2)),1);
latent_sz = [opts.patchsize/2^N_pool,opts.patchsize/2^N_pool,N_filters*2^N_pool,opts.batchsize];
z0 = single(gpuArray(randn([size(x0,1)/2^N_pool,size(x0,2)/2^N_pool,N_filters*2^N_pool])));

for iter = 1:opts.N_iter
  
  xb = im2row_patch_sample_2D(x0,opts.patchsize,dataratio);
  xb = xb.';
  xb = reshape(xb,opts.patchsize,opts.patchsize,opts.N_channels,[]);
  x_real = xb(:,:,:,1:opts.batchsize);
%   x_real = x0;
  
  %update D
  D.eval({'images',x_real},'forward');
  pred = D.getValue('pred');
  D_loss_real = sum((1-pred(:)).^2);
  D_der = 2*(pred-1);

  D.eval({'images',x_real},'backward',D_der);
  
%   D.displayVars
  D_state = accumulateGradientsAutoNN(D, D_state, D_params, opts.batchsize, []) ;
%   clip_weights(D,opts.clip);
  
  latent = randn(latent_sz,'like',x_real);
  G.eval({'latent',latent},'forward');
  x_fake = G.getValue('x');
  D.eval({'images',x_fake},'forward');
  pred = D.getValue('pred');
  D_loss_fake = sum((pred(:)).^2);
  D_der = 2*(pred);
  D.eval({'images',x_real},'backward',D_der);
  D_state = accumulateGradientsAutoNN(D, D_state, D_params, opts.batchsize, []) ;
%   clip_weights(D,opts.clip);
  
  D_loss = D_loss_real + D_loss_fake;
  fprintf('iter = %d, real val = %10.3e, fake val = %10.3e\n',iter,D_loss_real,D_loss_fake);
  
  %update G
  dataratio = opts.batchsize/(size(x0,1)*size(x0,2)) + 1e-4;
  xb = im2row_patch_sample_2D(x0,opts.patchsize,dataratio);
  xb = xb.';
  xb = reshape(xb,opts.patchsize,opts.patchsize,opts.N_channels,[]);
  x_real = xb(:,:,:,1:opts.batchsize);
%   x_real = x0;
%   latent = randn(latent_sz,'like',x_real);
  G.eval({'latent',latent},'forward');
  x_fake = G.getValue('x');
  D.eval({'images',x_fake},'forward');
  pred = D.getValue('pred');
  G_loss = sum((1-pred(:)).^2);
  G_der = 2*(pred-1);
  D.eval({'images',x_fake},'backward',G_der);
  G_der = D.getDer('images');
  G.eval({'latent',latent},'backward',G_der);
%   G.displayVars
%   D.displayVars
  G_state = accumulateGradientsAutoNN(G, G_state, G_params, opts.batchsize, []) ;

  G_loss_hist = [G_loss_hist,gather(G_loss)];
  D_loss_hist = [D_loss_hist,gather(D_loss)];
  
  if ~mod(iter,100)
    G.eval({'latent',z0},'forward');
    x_fake = G.getValue('x');
    subplot(131);
    imshow(x_fake(:,:,:,1))
    drawnow;
    subplot(132);
    imshow(x_real(:,:,:,1))
    drawnow;
    subplot(133);
    hold off;
    plot(G_loss_hist(:))
    hold on;
    plot(D_loss_hist(:));
%     legend('G','D');
    drawnow;
  end
  
  if ~mod(iter,100)
    
  	imwrite(gather(x_fake),[write_dir,num2str(iter),'.jpg'],'Quality',100);
  end
  
end



