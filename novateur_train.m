function novateur_train(dataset, modelfilename)
%% novateur_train(dataset, feattype)
%% dataset could be: KITTY, VOC2007
%% modelfilename is the file that store the trained model
p1 = mfilename('fullpath');
pos = strfind(p1,'/');
p1 = p1(1:pos(end));
addpath(genpath([p1 '/liblinear-1.93']));
addpath(genpath([p1 'gmm']));
switch dataset
    case 'KITTY'
        classnum = 8;
        datapath_samp2 = '/home/song/codepackage/project/sali_net/imdb/cache';
        neg_class = 10;
        cache_file_prefix = [datapath_samp2 '/datadb_cnn_kitty_train_c'];
        featdim = 256;
    case 'VOC2007'
        classnum = 20;
        datapath_samp2 = '/home/song/codepackage/project/sali_net/imdb/cache';
        cache_file_prefix = [datapath_samp2 '/datadb_cnn_voc_2007_train_c'];
        neg_class = 21;
        featdim = 256;
end
model_ggdsvm_p5 = cell(classnum,1);
model_svm_p5 = cell(classnum,1);
accuracy = zeros(classnum,2);

for( class = 1 : classnum )
    disp(['processing training phase ' num2str(class) '/' num2str(classnum)]); 
    % load training data
    cache_file = [cache_file_prefix num2str(class)];
    data1 = load(cache_file);
    cache_file = [cache_file_prefix num2str(neg_class)];
    data2 = load(cache_file);
    
    % prepare training and testing data
    datadb_train.feat = zeros(featdim, 5, 5, 20000);
    datadb_train.feat(:,:,:,1:10000) = data1.feat;
    datadb_train.feat(:,:,:,10001:20000) = data2.feat;
    datadb_train.flag = [ones(10000, 1); zeros(10000, 1)];
    
    data_train = reshape(datadb_train.feat(:,:,:,[1:5000, 10001:15000]), [featdim, 5, 5, 10000]);
    flag_train = [ones(5000,1);zeros(5000,1)];
    data_test = reshape(datadb_train.feat(:,:,:,[5001:10000, 15001:20000]), [featdim, 5, 5 10000]);
    flag_test = [ones(5000,1);zeros(5000,1)];
    c = 0.25;

    % training: fit the GGD
    feature = reshape(data_train, [featdim, 25*10000]);
    [alpha1, beta1, mean1] = fitGMM(feature(:,1:25*5000),7,1.5);
    [alpha2, beta2, mean2] = fitGMM(feature(:,1+25*5000:250000),7,1.5);
    model_ggdsvm_p5{class,1}.alpha1 = alpha1;
    model_ggdsvm_p5{class,1}.beta1 = beta1;
    model_ggdsvm_p5{class,1}.mean1 = mean1;
    model_ggdsvm_p5{class,1}.alpha2 = alpha2;
    model_ggdsvm_p5{class,1}.beta2 = beta2;
    model_ggdsvm_p5{class,1}.mean2 = mean2;
    % training: training svm model after ggd
    Y3 = reshape(data_test, [featdim, 25*10000]);
    Y3 = probGMM(alpha1,beta1,Y3,7, mean1) - probGMM(alpha2,beta2,Y3,7, mean2);
    Y3 = reshape(Y3, [featdim,5,5,10000]);
    temp1 = reshape(Y3(:,3,3,:), [featdim, 1, 10000]);
    temp2 = reshape(Y3(:,2:4,2:4,:), [featdim, 9, 10000]);
    temp3 = reshape(Y3(:,:,:,:), [featdim, 25, 10000]);
    data_test2 = [reshape(mean(temp1,2),[featdim,10000]);reshape(mean(temp2,2),[featdim,10000]);reshape(mean(temp3,2),[featdim,10000])];
    model_ggdsvm_p5{class,1}.model = train(flag_test, sparse(data_test2), ['-s 2 -c ' num2str(c)], 'col');
    % testing   
    Y3 = reshape(data_test, [featdim, 25*10000]);
    Y3 = probGMM(alpha1,beta1,Y3,7, mean1) - probGMM(alpha2,beta2,Y3,7, mean2);
    Y3 = reshape(Y3, [featdim,5,5,10000]);
    temp1 = reshape(Y3(:,3,3,:), [featdim, 1, 10000]);
    temp2 = reshape(Y3(:,2:4,2:4,:), [featdim, 9, 10000]);
    temp3 = reshape(Y3(:,:,:,:), [featdim, 25, 10000]);
    data_test4 = [reshape(mean(temp1,2),[featdim,10000]);reshape(mean(temp2,2),[featdim,10000]);reshape(mean(temp3,2),[featdim,10000])];
    [predicted_label, temp, decision_values] = predict(flag_test, sparse(data_test4), model_ggdsvm_p5{class,1}.model, '','col');
    accuracy(class,1) = temp(1);
    
    % training: training svm model
    Y3 = data_train;
    temp1 = reshape(Y3(:,3,3,:), [featdim, 1, 10000]);
    temp2 = reshape(Y3(:,2:4,2:4,:), [featdim, 9, 10000]);
    temp3 = reshape(Y3(:,:,:,:), [featdim, 25, 10000]);
    data_test2_3 = [reshape(mean(temp1,2),[featdim,10000]);reshape(mean(temp2,2),[featdim,10000]);reshape(mean(temp3,2),[featdim,10000])];
    model_svm_p5{class,1}.model = train(flag_test, sparse(data_test2_3), ['-s 2 -c ' num2str(c)], 'col');
    % testing
    Y3 = data_test;
    temp1 = reshape(Y3(:,3,3,:), [featdim, 1, 10000]);
    temp2 = reshape(Y3(:,2:4,2:4,:), [featdim, 9, 10000]);
    temp3 = reshape(Y3(:,:,:,:), [featdim, 25, 10000]);
    data_test4_3 = [reshape(mean(temp1,2),[featdim,10000]);reshape(mean(temp2,2),[featdim,10000]);reshape(mean(temp3,2),[featdim,10000])];
    [predicted_label, temp, decision_values] = predict(flag_test, sparse(data_test4_3), model_svm_p5{class,1}.model, '','col');
    accuracy(class,2) = temp(1);
    save(modelfilename,'model_ggdsvm_p5','model_svm_p5');
end
end