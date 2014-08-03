function [ salmap ] = ...
    local2global_object_saliency_superpatch8( filenames, fg_ratio, bg_ratio )
%detecting image saliency by using superpatch acceleration
%8 connectivity
if nargin ==1
    fg_ratio = 0.1;
    bg_ratio = 0.7;
end

%default parameters
% fg_ratio=0.1;%ratio for fg patches
% bg_ratio=0.7;%ratio for bg patches

% prior:
% 1. contrast consistency (SCC)
% 2. color prior (CP)
% 3. spatial weighting (SW): central weighting

%%
%parameters
cc_nn=8;%min with # of coarse patch
spasigma=200;%for spatial weighting %%%parameter
corsigma=1;
verbose=0;
%0 for viewing nothing
%1 for displaying the processing message
%2 for view the salient result only

salmap = cell(length(filenames), 1);
for ii=1: length(filenames)
    if verbose == 1
        fprintf(1, 'load: %s\n', filenames{ii,1});
    end
    ori_im = im2double(imread( filenames{ii,1}));%read the original image scale
    [ori_h, ori_w, ori_c] = size(ori_im);
    
    %Lab color space
    [im1, im2, im3]=RGB2Lab(ori_im);
    %additional matlab file of RGB2Lab
    
    fim(:,:,1)=mat2gray(im1);%feature image
    fim(:,:,2)=mat2gray(im2);
    fim(:,:,3)=mat2gray(im3);
    
    %%
    %generate finer layer feature representation
    fparams=[];
    fparams.pwidth=2;
    [fpmat, fpcmat, foparams] = ...%finer patches
        patch_gen(fim, fparams);

    fnum_patch=foparams.num_patchx*foparams.num_patchy;
    fpmatim = reshape([1:1: fnum_patch], ...
        foparams.num_patchy, foparams.num_patchx);
    %pmatim is the feature image
    
    %CP
    cpim = fim(:,:,1);%use L channel for color prior
    %normalize it into unit mat
    [ppmat]=patch_gen(cpim, fparams);
    
    %%
    %use super-patches representation
    cparams=[];
    cparams.pwidth=1;
    [cpmat, ~, coparams] = ...%coarse patches
        patch_gen(fpmatim, cparams);
    cnum_patch=coparams.num_patchx*coparams.num_patchy;
    %num of coarse patches

    %convert the element from normal patches into super patches
    cpcel = cell(cnum_patch,1);%superpatch color mat
    ccmat = zeros(cnum_patch,6);%superpatch coordinate matrix
    ppcel = cell(cnum_patch,1);%superpatch color prior
    for pp=1:cnum_patch
        ind=cpmat(pp,:);
        sfea=fpmat(ind,:);
        pfea=ppmat(ind,:);
        cpcel(pp,1)={sfea(:)'};
        ppcel(pp,1)={pfea(:)'};
        ccmat(pp,[1,2]) = fpcmat( ind(1,5), [1,2]);%for 8-connectivity
        ccmat(pp,[3,4]) = fpcmat( ind(1,1), [3,4]);%for 8-connectivity
        ccmat(pp,[5,6]) = fpcmat( ind(1,9), [5,6]);%for 8-connectivity
    end
    %CP
    cppmat=exp(mean(cell2mat(ppcel),2));%prior patch matrix
    ppcel=[];
    
    %SC
    cpdist=pdist(cell2mat(cpcel), 'euclidean');%color distances for patches
    ccdist=pdist( ccmat(:,[1:2]), 'euclidean');%spatial distances for patches
    sc_dist=squareform(cpdist ./ (1+ 1*ccdist));
    msc_dist=mean(sc_dist,2);%take their average distance
    cpcel=[];
    
    %SCC
    cpdistmat=squareform(cpdist);
    cstmap=msc_dist;%need a predefined contrast map
    knn=min(cc_nn,cnum_patch);%%%%parameters
    [scpdistmat, scpdistind]=sort(cpdistmat, 2, 'ascend');
    %sort with their (pi to pj) color similarity 
    %(i.e. knn=1 is the patch itself)
    colind=scpdistind(:,1:knn);%NN patch indices
    colorsimwei=exp(-scpdistmat(:,1:knn)./corsigma);%to similarity
    normA=sum(colorsimwei,2);
    normA = repmat(normA, 1, knn);    
    ncolorsimwei=colorsimwei./ normA;
    cc_dist=sum(cstmap(colind) .* ncolorsimwei,2);%consistence distance
        
    %SW
    %spatial weighting: weight more when the patches are located at the goldon areas.
    gspa=[0.5,0.5];
    gsloc=[ori_w*gspa(:,1), ori_h*gspa(:,2)];
    cgdist=pdist2(ccmat(:,[1:2]),gsloc, 'euclidean');
    cmindist=min(cgdist,[],2);%coarse minimum distance
    cmprob=exp(-cmindist/spasigma);


    mppdist=cc_dist.*cppmat.*cmprob;

    %filter out the out of box [0,1]
    mppdist(mppdist>1)=1;
    mppdist(mppdist<0)=0;
    
    %transfer the prior probability into the finer patches
    fplmat=zeros(fnum_patch,1);%foreground probability
    for pp=1:cnum_patch
        ind=cpmat(pp,:);
        fplmat(ind(:),1)=mppdist(pp,1);
    end
    bplmat = 1- fplmat;%background probability matrix
        
    %fg_ratio: most salient region as foreground
    %bg_ratio: less salient region as background
    %others: undetermine
    num1 = ceil(cnum_patch*fg_ratio);
    num2 = ceil(cnum_patch*(1-bg_ratio));
    smppdist=sort(mppdist, 'descend');
    cfg=mppdist >= smppdist(num1,1);%coarse foreground
    cbg=mppdist <= smppdist(num2,1);%coarse background
    
    flind=cpmat(cfg,:);%foreground index
    blind=cpmat(cbg,:);%background index

    fgmats=fpmat(flind(:),:);%foreground color mat
    fgcmat = fpcmat(flind(:), [1,2]);%foreground spatial mat
    bgmats=fpmat(blind(:),:);%background color mat
    bgcmat = fpcmat(blind(:), [1,2]);%background spatial mat
 
    f2pmat_col=exp(-pdist2(fpmat, fgmats, 'euclidean')/corsigma);
    b2pmat_col=exp(-pdist2(fpmat, bgmats, 'euclidean')/corsigma);
    
    f2pmat_spa=exp(-pdist2(fpcmat(:,[1,2]), fgcmat, 'euclidean')/spasigma);
    b2pmat_spa=exp(-pdist2(fpcmat(:,[1,2]), bgcmat, 'euclidean')/spasigma);

    f2cpmat = mean(f2pmat_col,2);%the average distance to the foreground patches
    b2cpmat = mean(b2pmat_col,2);%the average distance to the background patches

    f2spmat = mean(f2pmat_spa,2);
    b2spmat = mean(b2pmat_spa,2);

    upcmat=f2spmat.*f2cpmat.*fplmat;    
    fusdist= upcmat ./ (upcmat +b2spmat.*b2cpmat.*bplmat);%bayesian spatial distance

    salp=reshape(fusdist, foparams.num_patchy, foparams.num_patchx);
    salp_fr=mat2gray(imresize(salp,[ori_h,ori_w], 'bicubic'));
    %full resolution
    salmap(ii,1)={ salp_fr };
    if verbose == 2
        figure, imshow([ori_im, salp_fr]);
    end
end
end

