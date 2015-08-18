function novateur_test_new(dataset, modelfilename, norm)
%% novateur_test(dataset, feattype, traintest, class, norm)
%% dataset: KITTY, VOC2007. VOC2008_1023
%% norm: normalize the saliency map (non zero value) or not (zero)
p1 = mfilename('fullpath');
pos = strfind(p1,'/');
p1 = p1(1:pos(end));
addpath(genpath([p1 '/liblinear-1.93']));
addpath(genpath([p1 'gmm']));
save_prefix = fullfile('/data/Can/sal',dataset);
mkdir(save_prefix);
addpath('/data/Can/song_code/Novateur/matlab/liblinear/liblinear-2.01/matlab');
%savepathprefix=fullfile(save_prefix,'salimap');
%mkdir(savepathprefix);
switch(dataset)
    case 'KITTY'
        classnum = 8;
        datasetpath = '/home/song/dataset2/song_KITTY/training';
        imgpath = [datasetpath '/image_2'];
        image_ids = textread([datasetpath '/data_new/sublist/val.txt'], '%s');
        boxpath = [datasetpath '/data_new/box'];
        featpath = [datasetpath '/cnn_bfrec'];
        %savepathprefix = [datasetpath '/salimap/c'];
        save_prefix = fullfile(save_prefix,dataset);
        if(~exist(save_prefix,'dir'))
            mkdir(save_prefix);
        end
        savepathprefix = fullfile(save_prefix,'salimap');
        if(~exist(savepathprefix,'dir'))
            mkdir(savepathprefix);
        end
        savepathprefix = fullfile(savepathprefix,'c');
        if(~exist(savepathprefix,'dir'))
            mkdir(savepathprefix);
        end
        %savepathprefix = [datasetpath '/salimap/c'];
        for class = 1 : classnum
            savepath = [savepathprefix num2str(class) '/'];
            system(['mkdir ' savepath]);
        end
        if(isempty(modelfilename))
            load('model_KITTY.mat','model_ggdsvm_p5');
        else
            load(modelfilename,'model_ggdsvm_p5');
        end
        imgsuffix = 'png';
    case 'VOC2007'
        classnum = 20;
        datasetpath = '/home/song/dataset2/VOC2007';
        imgpath = [datasetpath '/VOCdevkit/VOC2007/JPEGImages'];
        image_ids = textread([datasetpath '/VOCdevkit/VOC2007/val.txt'], '%s');
        boxpath = [datasetpath '/box'];
        featpath = [datasetpath '/cnn/JPEGImages'];
        savepathprefix = [datasetpath '/salimap/c'];
        for class = 1 : classnum
            savepath = [savepathprefix num2str(class) '/'];
            system(['mkdir ' savepath]);
        end
        if(isempty(modelfilename))
            load('model_KITTY.mat','model_ggdsvm_p5');
        else
            load('model_VOC2007.mat','model_ggdsvm_p5');
        end
        imgsuffix = 'jpg';
    case 'VOC2008_1023'
        classnum = 20;
        datasetpath = '/data/Song/song/dataset/VOC2008_1023';
        imgpath = [datasetpath '/Data'];
        image_ids = textread([datasetpath '/namelist.txt'], '%s');
        featpath = [datasetpath '/cnn'];
        savepathprefix = [datasetpath '/salimap/c'];
        for class = 1 : classnum
            savepath = [savepathprefix num2str(class) '/'];
            system(['mkdir ' savepath]);
        end
        if(isempty(modelfilename))
            load('model_KITTY.mat','model_ggdsvm_p5');
        else
            load('model_VOC2007.mat','model_ggdsvm_p5');
        end
        imgsuffix = 'jpg';
      case 'New_data'
        classnum = 8;
        datasetpath ='/data/Can/NewData';% '/data/Song/song/dataset/VOC2008_1023';
        imgpath = [datasetpath '/Seq1'];
        image_ids = textread([datasetpath '/Seq1_list'], '%s');
        featpath = [datasetpath '/Seq1_fea'];
        savepathprefix = [datasetpath '/salimap'];
        mkdir(savepathprefix);
        savepathprefix = [savepathprefix '/Seq1'];
        mkdir(savepathprefix);
        for class = 1 : classnum
            savepath = [savepathprefix  '/c' num2str(class)];
            system(['mkdir ' savepath]);
        end
        savepathprefix=[savepathprefix  '/c'];
        if(isempty(modelfilename))
            load('model_KITTY.mat','model_ggdsvm_p5');
        else
            load('model_KITTY.mat','model_ggdsvm_p5');
        end
        imgsuffix = 'jpg';
end

for(n = 1 : length(image_ids))
    tic_toc_print('producing saliency map (%s): %d/%d\n', image_ids{n}, n, length(image_ids));
    img_name = image_ids{n}; 
    sl = length(img_name);
    if(sl>4 && strcmp(img_name(sl-2:sl),imgsuffix))
        img_name = img_name(1:sl-4);
    end
    featname = [featpath '/'  img_name '_CNN_conv5_feature.mat'];
    imagename = [imgpath '/' img_name '.' imgsuffix];
    img = imread(imagename);
    feat_cur = extract_cnn(img, featname, 'conv5');
    xlen = length(unique(feat_cur.x));
    ylen = length(unique(feat_cur.y));
    flag = zeros(ylen, xlen);
    
    for(class = 1 : classnum)
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
            save([savepathprefix num2str(class) '/' img_name '_salmap_u.mat'],'salimap');
            imwrite(salimap/3, [savepathprefix num2str(class) '/' img_name '_salmap_u.jpg']);
        else 
            sali_map_large2 = mat2gray(sali_map_large2).^4;
            imwrite(sali_map_large2, [savepathprefix num2str(class) '/' img_name '_salmap.jpg']);
        end
    end
end
end