function im = standarizeImage(im)

im = im2single(im);
if size(im,1) > 480, im = imresize(im, [480 NaN]);
end

% 이미지 type을 single로 변경
% 이미지 double을 single로 만든다.
% 이미지의 크기가 480을 넘으면 resize 한다.
