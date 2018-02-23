function y = periodic_conv(x,w,b,varargin)

if isa(x,'Layer')
  if numel(varargin) %set w and b
    opts.hasBias = false;
    opts.size = [];
    opts = vl_argparse(opts,varargin);
    if ~numel(opts.size)
      error('must provide size');
    end
      
    scale = sqrt(2 / prod(opts.size(1:3))) ;
    w = Param('value', randn(opts.size, 'single') * scale);
    
    if opts.hasBias
      b = Param('value',zeros(opts.size(4),1,'single'));
    else
      b = [];
    end
  end
  
  numInputDer = 1 + isa(w,'Layer') + isa(b,'Layer'); % count number differentiable layers
  y = Layer(@periodic_conv,x,w,b);
  y.numInputDer = numInputDer;
else %forward pass
  padsize = (size(w,1) - 1)*[1 1];
  y = padarray(x,padsize,'post','circular');
  y = vl_nnconv(y,w,b);
end