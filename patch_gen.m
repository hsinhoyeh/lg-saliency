function [ pmat, pcmat, outparams ] = ...
    patch_gen( fim, params )

%input:
%fim: the specified image (multi-channel for color, single-channel for feature image(coordinate))
%params: the used parameters

%when im is single channel, used to generate label patches

%output:
%pmat: patch matrix where each element is color feature
%#patchs by #dims by #channels
%pcmat: patch matrix where each element is spatial coordinate

pwidth = params.pwidth;% patch_width*2+1 is the patch size
% colspace = params.col;%the color space used

[ori_h, ori_w, ori_c] = size(fim);

pw=pwidth;%the width of a patch
pstepw=pw*2+1;%step for horizontal direction
ph=pwidth;%the height of a patch
psteph=ph*2+1;%step for vertical direction
    
num_dim = (pw*2+1)*(ph*2+1);%num of patch in total


% if ori_c > 1 %multiple channels
%     [im1, im2, im3]=...
%         colorspace_gen(im, colspace);
%     fim = ... %feature image
%         { mat2gray(im1) ...
%         , mat2gray(im2), ...
%           mat2gray(im3) };
% else
%     fim = {im};
% end



%construct patch coordinates
coordi_x = [pw+1:pstepw:ori_w-pw];
coordi_y = [ph+1:psteph:ori_h-ph];
[px, py] = meshgrid(coordi_x, coordi_y);

num_patchx = size(px,2);
num_patchy = size(px,1);
num_patch = num_patchx* num_patchy;

px1 = px(:)-pw;px2 = px(:)+pw;
py1 = py(:)-ph;py2 = py(:)+ph;


if ori_c > 1 %multiple channels
    mpcel = cell(1, ori_c);
    for cchen=1: ori_c
        mpmat = zeros(num_dim, num_patch);
        aim = fim(:,:,cchen);%current use image
        for pind=1: num_patch
            patch = aim( ...
                py1(pind,1): py2(pind,1), ...
                px1(pind,1): px2(pind,1));    
            mpmat(:,pind) = patch(:);
        end
        mpcel(1,cchen)={mpmat};
    end
    %concate the features from multiple dimensional
    mpmat=cat(1, mpcel{1,:});
    
else %use center pixel as its value
    mpmat = zeros(num_dim, num_patch);%patch matrix
    aim = fim;%current use image
    for pind=1: num_patch
        patch = aim( ...
            py1(pind,1): py2(pind,1), ...
            px1(pind,1): px2(pind,1));
        mpmat(:,pind) = patch(:);
    end
end

    
pcmat = [ px(:), py(:), px1, py1, px2, py2 ];
%cnt_x, cnt_y, left, top, right, bottom
pmat=mpmat';
outparams.num_patchx = num_patchx;
outparams.num_patchy = num_patchy;
outparams.im_w = ori_w;
outparams.im_h = ori_h;
outparams.num_dim=num_dim;
outparams.num_patch=num_patch;
end

