function [accuracy] = phow_caltech101()
% Originated from VLFEAT Application
% modified by wjhwang for Computer Vision Final Assignment
% Baseline method for caltech-101

%configure 을 정의한다.
conf.calDir = 'data/caltech-101' ; %image data가 있는 위치
conf.dataDir = 'data/' ;
conf.resultDir = 'result/';  %result data가 저장될 위치
conf.autoDownloadData = true ;

conf.numTrain = 15 ; % train image 수 = 15개
conf.numTest = 15 ; % test image 수 = 15개
conf.numClasses = 102 ; % 이미지를 가져올 폴더 수 102개 (카테고리)

conf.numWords = 600 ;
conf.numSpatialX = [2 4] ;
conf.numSpatialY = [2 4] ;
conf.quantizer = 'kdtree' ;
conf.svm.C = 10 ;

conf.svm.solver = 'sdca' ;
%conf.svm.solver = 'sgd' ;
%conf.svm.solver = 'liblinear' ;

conf.svm.biasMultiplier = 1 ;
conf.phowOpts = {'Step', 3} ;
conf.clobber = false ;
conf.tinyProblem = false; %true ;
conf.prefix = 'baseline' ;
conf.randSeed = 1 ;

if conf.tinyProblem
  conf.prefix = 'tiny' ;
  conf.numClasses = 5 ;
  conf.numSpatialX = 2 ;
  conf.numSpatialY = 2 ;
  conf.numWords = 300 ;
  conf.phowOpts = {'Verbose', 2, 'Sizes', 7, 'Step', 5} ;
end

conf.vocabPath = fullfile(conf.resultDir, [conf.prefix '-vocab.mat']) ;
% vocabulary data 저장 위치 
conf.histPath = fullfile(conf.resultDir, [conf.prefix '-hists.mat']) ;
% histogram data 저장
conf.modelPath = fullfile(conf.resultDir, [conf.prefix '-model.mat']) ;
% model 정보 저장
conf.resultPath = fullfile(conf.resultDir, [conf.prefix '-result']) ;
% 결과 저장

randn('state',conf.randSeed) ;
rand('state',conf.randSeed) ;
vl_twister('state',conf.randSeed) ;

% --------------------------------------------------------------------
%                                            Download Caltech-101 data
% --------------------------------------------------------------------

if ~exist(conf.calDir, 'dir') || ...
   (~exist(fullfile(conf.calDir, 'airplanes'),'dir') && ...
    ~exist(fullfile(conf.calDir, '101_ObjectCategories', 'airplanes')))
  if ~conf.autoDownloadData
    error(...
      ['Caltech-101 data not found. ' ...
       'Set conf.autoDownloadData=true to download the required data.']) ;
  end
  vl_xmkdir(conf.calDir) ;
  calUrl = ['http://www.vision.caltech.edu/Image_Datasets/' ...
    'Caltech101/101_ObjectCategories.tar.gz'] ;
  fprintf('Downloading Caltech-101 data to ''%s''. This will take a while.', conf.calDir) ;
  untar(calUrl, conf.calDir) ;
end

if ~exist(fullfile(conf.calDir, 'airplanes'),'dir')
  conf.calDir = fullfile(conf.calDir, '101_ObjectCategories') ;
end

% make the result directory
vl_xmkdir(conf.resultDir) ;

% --------------------------------------------------------------------
%  디렉토리에서 이미지를 불러온다.                                 Setup data
% --------------------------------------------------------------------
classes = dir(conf.calDir) ;
classes = classes([classes.isdir]) ;
classes = {classes(3:conf.numClasses+2).name} ;
%이미지 디렉토리에서 첫 번째부터 세 번째까지 디렉토리를 classes에 넣는다.

images = {} ; % 모든 이미지들의 인덱스가 들어간다.
imageClass = {} ; % 모든 이미지의 카테고리가 들어간다
for ci = 1:length(classes)
  ims = dir(fullfile(conf.calDir, classes{ci}, '*.jpg'))' ;
  ims = vl_colsubset(ims, conf.numTrain + conf.numTest) ;
  ims = cellfun(@(x)fullfile(classes{ci},x),{ims.name},'UniformOutput',false) ;
  images = {images{:}, ims{:}} ;
  imageClass{end+1} = ci * ones(1,length(ims)) ;
end
selTrain = find(mod(0:length(images)-1, conf.numTrain+conf.numTest) < conf.numTrain) ;
% 16~30...selTest 인덱스를 제외한 15*102개의 이미지의 인덱스가 들어간다.
selTest = setdiff(1:length(images), selTrain) ;
% 16~30 인덱스 15개의 테스트 이미지의 인덱스가 들어간다.
imageClass = cat(2, imageClass{:}) ;

model.classes = classes ;
model.phowOpts = conf.phowOpts ;
model.numSpatialX = conf.numSpatialX ;
model.numSpatialY = conf.numSpatialY ;
model.quantizer = conf.quantizer ;
model.vocab = [] ;
model.w = [] ;
model.b = [] ;
%model.classify = @classify ;

% --------------------------------------------------------------------
%  vocab 만드는 과정                                   Train vocabulary
% --------------------------------------------------------------------
% vl_phow() 함수를 사용해서 descre를 추출했다. 특징점을 BoVW를 만들기 위해 kmeans 클러스터링을 해야한다.
% k-means 클러스터링이란 k개의 센터를 정하고 센터에서 가까운 데이터들을 하나의 집합으로 만드는 것

if ~exist(conf.vocabPath) || conf.clobber

  % Get some PHOW descriptors to train the dictionary
  selTrainFeats = vl_colsubset(selTrain, 100) ;
  % x(selfTrain)가 n(30)의 개수보다 작으면 랜덤이 아닌 컬럼을 그대로 반영하지만
  % x의 컬럼이 더 많다면 랜덤으로 n의 개수로 랜덤 반영한다.
  % 15개 중 30개를 특징점 train image로 뽑았다. 
  descrs = {} ;
  % 128 x K matrix의 descriptors를 리턴 받을 장소
  
  %for ii = 1:length(selTrainFeats)
  parfor ii = 1:length(selTrainFeats)
    im = imread(fullfile(conf.calDir, images{selTrainFeats(ii)})) ;
    im = standarizeImage(im) ;
    [drop, descrs{ii}] = vl_phow(im, model.phowOpts{:}) ;
  end
  % 30개의 이미지 인덱스를 사용해 im에 이미지를 하나하나 불러온 후 standarizeImage()함수로
  % 이미지를 같은 타입으로 만든다. (vfleat는 single form의 이미지를 요구함)
  % 그 후 vl_phow 함수를 사용해서 model.phowOpts{:} 옵션으로 im에 있는 이미지의
  % features(frames로 구성-frames(1:2,:)는 특징점의 x,y좌표 (센터 값)) 와 descriptor를 구함

  descrs = vl_colsubset(cat(2, descrs{:}), 10e4) ;
  descrs = single(descrs) ;

  % Quantize the descriptors to get the visual words
  vocab = vl_kmeans(descrs, conf.numWords, 'verbose', 'algorithm', 'elkan', 'MaxNumIterations', 50) ;
  % vl_kmeans 함수에 입력데이터 descre를 넣고, conf.numWords으로 클러스터링하고 elkan알고리즘을 사용해서
  % vocab에 저장
  save(conf.vocabPath, 'vocab') ;
  % 코드북(visual words) 만들기 위해 quantization한다.
  % vl_kmeans 함수를 사용해 센터값을 구한다.
  
else
  load(conf.vocabPath) ;
end

model.vocab = vocab ;

if strcmp(model.quantizer, 'kdtree')
  model.kdtree = vl_kdtreebuild(vocab) ;
end
% vl_kdtreebuild(x) x 데이터의 인덱스로 kd-tree forest를 만든다.


% --------------------------------------------------------------------
%  histogram을 만드는 과정                     Compute spatial histograms
% --------------------------------------------------------------------

if ~exist(conf.histPath) || conf.clobber
  hists = {} ; % histogram을 저장할 곳
  
  parfor ii = 1:length(images) % 모든 카테고리 images
  % for ii = 1:length(images)
    fprintf('Processing %s (%.2f %%)\n', images{ii}, 100 * ii / length(images)) ;
    im = imread(fullfile(conf.calDir, images{ii})) ;
    hists{ii} = getImageDescriptor(model, im); 
  end
  hists = cat(2, hists{:}) ;
  
  save(conf.histPath, 'hists') ;
else
  load(conf.histPath) ;
end

% --------------------------------------------------------------------
% hists로 커널 맵 구성 (linear SVM solver PEGASOS를 train하기 위해)    Compute feature map
% --------------------------------------------------------------------

psix = vl_homkermap(hists, 1, 'kchi2', 'gamma', .5) ;

% --------------------------------------------------------------------
%                                                            Train SVM
% --------------------------------------------------------------------

if ~exist(conf.modelPath) || conf.clobber
  switch conf.svm.solver
    case {'sgd', 'sdca'}
      lambda = 1 / (conf.svm.C *  length(selTrain)) ;
      w = [] ;
      parfor ci = 1:length(classes)
        perm = randperm(length(selTrain)) ;
        fprintf('Training model for class %s\n', classes{ci}) ;
        y = 2 * (imageClass(selTrain) == ci) - 1 ;
        [w(:,ci) b(ci) info] = vl_svmtrain(psix(:, selTrain(perm)), y(perm), lambda, ...
          'Solver', conf.svm.solver, ...
          'MaxNumIterations', 50/lambda, ...
          'BiasMultiplier', conf.svm.biasMultiplier, ...
          'Epsilon', 1e-3);
      end

    case 'liblinear'
      svm = train(imageClass(selTrain)', ...
                  sparse(double(psix(:,selTrain))),  ...
                  sprintf(' -s 3 -B %f -c %f', ...
                          conf.svm.biasMultiplier, conf.svm.C), ...
                  'col') ;
      w = svm.w(:,1:end-1)' ;
      b =  svm.w(:,end)' ;
  end

  model.b = conf.svm.biasMultiplier * b ;
  model.w = w ;

  save(conf.modelPath, 'model') ;
else
  load(conf.modelPath) ;
end

% --------------------------------------------------------------------
%                                                Test SVM and evaluate
% --------------------------------------------------------------------

% Estimate the class of the test images
scores = model.w' * psix + model.b' * ones(1,size(psix,2)) ;
[drop, imageEstClass] = max(scores, [], 1) ;

% Compute the confusion matrix for the test set (not training)
idx = sub2ind([length(classes), length(classes)], ...
              imageClass(selTest), imageEstClass(selTest)) ;
confus = zeros(length(classes)) ;
confus = vl_binsum(confus, ones(size(idx)), idx) ;

% Plots
figure(1) ; clf;
subplot(1,2,1) ;
imagesc(scores(:,[selTrain selTest])) ; title('Scores') ;
set(gca, 'ytick', 1:length(classes), 'yticklabel', classes) ;
subplot(1,2,2) ;
imagesc(confus) ;
accuracy = 100 * mean(diag(confus)/conf.numTest);
title(sprintf('Confusion matrix (%.2f %% accuracy)', accuracy)) ;
%print('-depsc2', [conf.resultPath '.ps']) ;
print('-dpdf', [conf.resultPath '.pdf']) ;
save([conf.resultPath '.mat'], 'confus', 'conf') ;
