function novateur_genesalimap(imgpath, featpath,classtype,  savepath,  norm,modelfilename)
%% novateur_genesalimap(imgpath, featpath,classtype,  savepath, norm)
%% generate saliency map for person and car classes
%% imgpath: input image path that stores images for calculating the saliency map
%% featpath: folder that stores the cnn features of the corresponding images in imgpath, if featpath==[], use the default path
%% classtype: 'person','car'
%% savepath: if savepath == [], use the default path
%% norm: normalize the saliency map (non zero value) or not (zero)
p1 = mfilename('fullpath');
pos = strfind(p1,'/');
p1 = p1(1:pos(end));
addpath(genpath([p1 '/liblinear-1.93']));
addpath(genpath([p1 'gmm']));

switch(classtype)
    case 'car'
        class = 1;
    case 'person'
        class = 4;
end
imglist = dir([imgpath '/*.png']);
imglist = [imglist; dir([imgpath '/*.jpg'])];
image_ids = {imglist.name};
image_ids = image_ids';
if(isempty(featpath))
    featpath = [imgpath '/cnn'];
end
%system(['mkdir ' featpath]);
if(isempty(savepath))
    savepath = [imgpath '/salimap'];
end
system(['mkdir ' savepath]); 
load(modelfilename, 'model_ggdsvm_p5');

for(n = 1 : length(image_ids))
    tic_toc_print('producing saliency map (%s): %d/%d\n', image_ids{n}, n, length(image_ids));
    img_name = image_ids{n}; 
    featname = [featpath '/'  img_name(1:end-4) '_CNN_conv5_feature.mat'];
    imagename = [imgpath '/' img_name];
    img = imread(imagename);
    feat_cur = extract_cnn(img, featname, 'conv5');
    xlen = length(unique(feat_cur.x));
    ylen = length(unique(feat_cur.y));
    flag = zeros(ylen, xlen);
         
    data_test_temp = reshape(feat_cur.feat, [256, ylen*xlen]);
    data_test_temp = probGMM(model_ggdsvm_p5{class}.alpha1,model_ggdsvm_p5{class}.beta1,data_test_temp,7, model_ggdsvm_p5{class}.mean1)...
         - probGMM(model_ggdsvm_p5{class}.alpha2,model_ggdsvm_p5{class}.beta2,data_test_temp,7, model_ggdsvm_p5{class}.mean2);
    data_test = zeros(256, ylen+4, xlen+4);
    data_test(:,3:ylen+2, 3:xlen+2) = reshape(data_test_temp,[256, ylen, xlen]);
    data_test(:, 1:2, 3:xlen+2) = repmat(data_test(:, 3, 3:xlen+2),[1,2,1]);
    data_test(:, ylen+3:ylen+4, 3:xlen+2) = repmat(data_test(:, ylen+2, 3:xlen+2),[1,2,1]);
    data_test(:,:, 1:2) = repmat(data_test(:,:, 3),[1,1,2]);
    data_test(:,:, xlen+3:xlen+4) = repmat(data_test(:,:, xlen+2),[1,1,2]);
    data_test0 = zeros(256, 5, 5, xlen*ylen);
    for( k1 = 1:5 )
        for( k2 = 1:5 )
            data_test0(:, k1, k2, : ) = reshape(data_test(:, k1: k1 + ylen - 1, k2: k2 + xlen - 1),[256,1,1,xlen*ylen]);
        end
    end
    temp1 = reshape(data_test0(:,3,3,:), [256, 1, xlen*ylen]);
    temp2 = reshape(data_test0(:,2:4,2:4,:), [256, 9, xlen*ylen]); 
    temp3 = reshape(data_test0(:,:,:,:), [256, 25, xlen*ylen]);
    data_test0 = [reshape(mean(temp1,2),[256,xlen*ylen]);reshape(mean(temp2,2),[256,xlen*ylen]);reshape(mean(temp3,2),[256,xlen*ylen])];   
        
     
    [predicted_label, accuracy, decision_values] = predict(flag(:), sparse(data_test0), model_ggdsvm_p5{class, 1}.model, '','col');
    sali_map2 = reshape(decision_values, [ylen, xlen]);
    sali_map_large2 = imresize(sali_map2, [feat_cur.hgt, feat_cur.wid]);
        
    % normalize the saliency map or not
    if(norm==0)
        salimap = sali_map_large2;
        save([savepath img_name '_salmap_u.mat'],'salimap');
        imwrite(salimap/3, [savepath '/' img_name '_salmap_u.jpg']);
    else 
        sali_map_large2 = mat2gray(sali_map_large2).^4;
        imwrite(sali_map_large2, [savepath '/' img_name '_salmap.jpg']);
    end
end
end