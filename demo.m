%% demo for training 
dataset = 'KITTY';% could be : 'VOC2007','KITTY'
modelfilename = 'model_VOC2007_new.mat';
novateur_train(dataset, modelfilename);

%% demo for generating saliency map for the testing set
dataset = 'KITTY';% 'KITTY' for model trained on 'KITTY', 'VOC2007' or 'VOC2008_1023' for model trained on ''VOC2007'
modelfilename = 'model_VOC2007_new.mat';
norm = 1;
%novateur_test(dataset, modelfilename, norm);

%% demo for generating saliency map for any image set for the 'car' or 'person' dataset
carsets1 = {'salivideo/0001', 'salivideo/0004', 'salivideo/0007', 'salivideo/0002', 'salivideo/0011'};
carsets2 = {'UGV/20130902.190148.751_Cam0', 'UGV/20131102.195434.772_Cam0', 'UGV/20131102.192756.224'};
personsets = {'salivideo/0021', 'salivideo/15'};
for(n = 1 : length(carsets1))
    imgpath = ['/data/Song/song/dataset/' carsets1{n} '/img'];
    featpath = ['/data/Song/song/dataset/' carsets1{n} '/cnn'];
    classtype = 'car';
    norm = 1;
    savepath = [];
    modelfilename = 'model_VOC2007_new.mat';
    novateur_genesalimap(imgpath, featpath,classtype,  savepath,  norm,modelfilename);
end
for(n = 1 : length(carsets2))
    imgpath = ['/data/imgDB/DB/' carsets2{n} '/img'];
    featpath = ['/data/imgDB/DB/' carsets2{n} '/cnn'];
    classtype = 'car';
    norm = 1;
    savepath = [];
    modelfilename = 'model_VOC2007_new.mat';
    novateur_genesalimap(imgpath, featpath,classtype,  savepath,  norm,modelfilename);
end
