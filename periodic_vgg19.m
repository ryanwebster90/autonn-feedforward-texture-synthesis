function x = periodic_vgg19()
net = load('vgg19_relu3_1.mat');


x = Input();
for l = [1,6]
  w = Param('value',single(net.layers{l}.weights{1}));
  b = Param('value',net.layers{l}.weights{2});
  
  x = periodic_conv(x,w,b);
  x = vl_nnrelu(x);
  
  w = Param('value',net.layers{l+2}.weights{1});
  b = Param('value',net.layers{l+2}.weights{2});
  x = periodic_conv(x,w,b);
  x = vl_nnrelu(x);
  
  x = vl_nnpool(x,[2 2],'method','avg','stride', [2 2],'pad',[0 1 0 1]);
end

w = Param('value',net.layers{end-1}.weights{1});
b = Param('value',net.layers{end-1}.weights{2});
x = periodic_conv(x,w,b);
x = vl_nnrelu(x);

% net = Net(x);
% net.move('gpu');