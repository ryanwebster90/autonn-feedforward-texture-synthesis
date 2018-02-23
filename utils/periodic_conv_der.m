function [dx,dw,db] = periodic_conv_der(x,w,b,dy)

padsize = (size(w,1)-1)*[1 1];

% dzdx of conv
y = padarray(x,padsize,'post','circular');
[dy,dw,db] = vl_nnconv(y, w, b, dy) ;

dx = pad_der(dy,padsize);


function dzdx = pad_der(dzdy,padsize)
i1 = 1:padsize(1);
i2 = padsize(1)+1:size(dzdy,1)-padsize(1);
i3 = size(dzdy,1)-padsize(1)+1:size(dzdy,1);

j1 = 1:padsize(2);
j2 = padsize(2)+1:size(dzdy,2)-padsize(2);
j3 = size(dzdy,2)-padsize(2)+1:size(dzdy,2);

dzdy(i1,j1,:,:) = dzdy(i1,j1,:,:) + dzdy(i3,j1,:,:) + dzdy(i1,j3,:,:) + dzdy(i3,j3,:,:);
dzdy(i2,j1,:,:) = dzdy(i2,j1,:,:) + dzdy(i2,j3,:,:);
dzdy(i1,j2,:,:) = dzdy(i1,j2,:,:) + dzdy(i3,j2,:,:);

dzdx = dzdy(1:size(dzdy,1)-padsize(1),1:size(dzdy,2)-padsize(2),:,:);