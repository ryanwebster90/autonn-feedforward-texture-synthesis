function [X,patch_inds,sample_inds] = im2row_patch_sample_2D(x,patchsize,dataratio,varargin)
% Uses linear 2D patch indices returned by create_patch_sample_inds to take
% patches from a tensor x (M by N by C). There is an
% emphasis on speed, especially when x is a single gpuArray. If your goal
% is to take patches from an image in some way, this will be much faster than
% most native or mex implementations.
%
% [patch_sample_inds,sample_inds] = create_patch_sample_inds(x,5,1);
% [X,patch_inds,sample_inds] = im2row_patch_sample_2D(x,patchsize,dataratio,varargin)
% Takes every (patchsize by patchsize by C) patch from x, by using
% indexing. For example patch_inds contain indices of patches in 2D, then
% these indices are replicated across channel dimension via  fast indexing 
% X = x(patch_inds(:),:), where the first two dimension of x were vectorized
% 
% See also,
% create_patch_sample_inds, row2im_patch_sample_2D, const_conv_periodic
%
%
% Copyright (C) 2017 Ryan Webster
% All rights reserved.
%
% This file is made available under the terms of the MIT license.

opts.patch_inds = []; % you can provide indices if you do not want to randomly sample the patches, e.g. a strided grid
opts = vl_argparse(opts,varargin);

x_sz = size(x);

if numel(opts.patch_inds)
  sample_inds = [];
  patch_inds = opts.patch_inds;
  patchsize = sqrt(size(patch_inds,2));
else
  [patch_inds,sample_inds] = create_patch_sample_inds(x,patchsize,dataratio);
end

% split first two dimension and remaining dimensions
X_sz = [size(patch_inds,1),patchsize^2*prod(x_sz(3:end))];
x = reshape(x,x_sz(1)*x_sz(2),[]);

% take patches from first 2D, replicate across all other dimensions
X = reshape( x(patch_inds(:),:),X_sz);