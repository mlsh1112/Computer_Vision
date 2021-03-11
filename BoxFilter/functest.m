im = imread('Lenna.png');
im = rgb2gray(im);
im = im2double(im);

sz=3;
kernel=ones(sz);
center=round(sz/2);

[height, width]=size(im);
[heightofKernel, widthofKernel]=size(kernel);

output=zeros(height,width);


for y=center:(height-(center-1))
   for x=center:(width-(center-1))
       for i=1:heightofKernel
           for j=1:widthofKernel
               output(y,x)=output(y,x)+kernel(i,j)*im(y+(i-center),x+(j-center));
           end
       end
   end
end

figure,imshow(output);