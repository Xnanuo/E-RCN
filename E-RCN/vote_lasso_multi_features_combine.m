function [save_name] = vote_lasso_multi_features_combine(feature_label,throld,repeats,KFold,save_folder,lambda_spe)
% clear;
% clc;
% close;
tic;
addpath('./utils'); % load utilities
addpath('./data');
addpath(genpath('./libsvm-3.21'));
features=feature_label.features;
label=feature_label.label;
save_folder=strcat(save_folder,'_',num2str(repeats))
if ~exist(save_folder)
    mkdir(save_folder) % 若不存在，在当前目录中产生一个子目录‘Figure’
end
fidname=strcat('UM_lasso_bn_246_',char(throld),'_',num2str(repeats),'.txt');
fide_path=fullfile(save_folder,fidname);
fid = fopen(fide_path, 'w');

for i=1:length(features)
    labels{1, i} = label;
    feature_lenght(i)=size(features{i},2);
end
% feature_label;
taskNums = length(features);


opts=[];
% Starting point
opts.init=2;            % starting from a zero point
% Termination criterion
opts.tol=1e-5; 
opts.tFlag=1;          % run .maxIter iterations
opts.maxIter=1e5;      % maximum number of iterations
% Mormalization
opts.nFlag=0;

%opts.rFlag=1; 
% lambdas =  [0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5 1 5  10 50 100 500 1000 5000 10000];
% lambdas =  [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 6 7 8 9 10];
weight=[8552,8276,8897];
weight_sum=25725;
lambdas=[1]
lambdas2 = [1];
lambdaInd = 1;
for lambda = lambdas
	lambda2Ind = 1;
	for lambda2 = lambdas2
		for t=1:taskNums  %特征数量
		featselectindex{lambdaInd,lambda2Ind,t}=zeros(feature_lenght(t),1);%每种特征246维
		feat_select_length{lambdaInd,lambda2Ind,t}=0;
        all_pre_label{lambdaInd,lambda2Ind,t}=zeros(repeats+1,size(features{1},1));
		temp= all_pre_label{lambdaInd,lambda2Ind,t};
		temp(repeats+1,:)=label(:,1);
		all_pre_label{lambdaInd,lambda2Ind,t}=temp;
        all_weight{lambdaInd,lambda2Ind,t}=zeros(feature_lenght(t),1);%246*tasknums
		end
		
		for r=1:repeats
			all_decision_values{lambdaInd,lambda2Ind,r}=zeros(size(labels{1},1),1);%最终分类器集成得到的预测概率
			all_pre_com_label{lambdaInd,lambda2Ind,r}=zeros(size(labels{1},1),1);%所有特征级联在一起分类
			
		end
		
		lambda2Ind = lambda2Ind + 1;
	end
	lambdaInd = lambdaInd + 1;
end

maxAcc = 0;
feature_length_best=0;
lambdaInd = 1;
for lambda = lambdas
	lambda2Ind = 1;
	for lambda2 = lambdas2
		for repeat=1:repeats
			%% K-fold cross valindation
			index = crossvalind('KFold', size(labels{1},1), KFold);
			for selectIndex=1:KFold
			   %% split training and testing dataset
				testIndex = (selectIndex==index);
				trainIndex = ~testIndex;
                for i=1:taskNums %每种特征分开训练svm
                    trainData{i} = features{i}(trainIndex, :);
                    testData{i} = features{i}(testIndex, :);
                    trainLabel{i} = labels{i}(trainIndex, :);
                    testLabel{i} = labels{i}(testIndex, :);
                   
				    [trainNum, d] = size(trainData{i});  % dimensionality.
				    [testNum, ~] = size(testData{i});
                %%feature select
                    [W, funVal2, ValueL2]= LeastC(trainData{i}, trainLabel{i}, lambda_spe(i), opts);
                
				    Weight = W';
                    Weight_abs=abs(W');
				    selectedTrainData_w= trainData{i};
				    selectedTestData_w = testData{i};
				    selectedTrainData{i} =selectedTrainData_w(:,Weight_abs(:)>(mean(Weight_abs(:))));
					selectedTestData{i} = selectedTestData_w(:,Weight_abs(:)>(mean(Weight_abs(:))));
                    
                    feat_select_length{lambdaInd,lambda2Ind,i}=feat_select_length{lambdaInd,lambda2Ind,i}+size(selectedTestData,2)/(KFold*repeats);
					
					overmeanindex=featselectindex{lambdaInd,lambda2Ind,i};%每种特征在所有实验中选中次数计数
					overmeanindex(Weight_abs(:)>(mean(Weight_abs(:))))=overmeanindex(Weight_abs(:)>(mean(Weight_abs(:))))+1;
					featselectindex{lambdaInd,lambda2Ind,i}=overmeanindex;
                    save_weight=all_weight{lambdaInd,lambda2Ind,i};
                %all_weight 是每类特征所有特征对应的权重
				    all_weight{lambdaInd,lambda2Ind,i}=save_weight+Weight_abs'/(KFold*repeats);
                end
                combine_decision_values=zeros(size(testLabel{1},1),1);
				for i=1:taskNums %每种特征训练一个SVM
				% 构建线性核矩阵
					ktrain{i} = selectedTrainData{i} * selectedTrainData{i}';
					Ktrain{i} = [(1:trainNum)', ktrain{i}];
					ktest{i} = selectedTestData{i} * selectedTrainData{i}';
					Ktest{i} = [(1:testNum)', ktest{i}];
					% SVM train and test
					SKmodel(i) = svmtrain(trainLabel{i}, Ktrain{i}, '-t 4 -b 1'); %#ok<*SVMTRAIN>
					[pre, acc, dec] = svmpredict(testLabel{i}, Ktest{i}, SKmodel(i), '-b 1');
		
					pre_label=all_pre_label{lambdaInd,lambda2Ind,i};%每种特征一个cell,每种特征在几个repeat下的预测值
					pre_label(repeat,testIndex)=dec(:,1);%这里的是概率,每个repeat下，会对每个被试做完一次预测，因此一个repeat刚好预测完全部数据
					all_pre_label{lambdaInd,lambda2Ind,i}=pre_label;

					decision_values=all_decision_values{lambdaInd,lambda2Ind,repeat};
					decision_values(testIndex)=decision_values(testIndex)+dec(:,1)/taskNums;%求了一个均值
                    %decision_values(testIndex)=decision_values(testIndex)+dec(:,1)*weight(i)/weight_sum;%求了一个weight的均值
					all_decision_values{lambdaInd,lambda2Ind,repeat}=decision_values;
					%貌似all_decision_values与combine_decision_values一样
					combine_decision_values=combine_decision_values+dec(:,1)/taskNums;
                    %combine_decision_values=combine_decision_values+dec(:,1)*weight(i)/weight_sum;
				end
				%% evaluating the model（对单组test评估）
				predictScore=combine_decision_values;
				predictLabel=predictScore;
				predictLabel(predictLabel>0.5)=1;
				predictLabel(predictLabel<=0.5)=-1;
				Acc= (sum(predictLabel(testLabel{1}==1)==1)+sum(predictLabel(testLabel{1}==-1)==-1))/testNum;
				Kfold_Acc(lambdaInd,lambda2Ind,(repeat-1)*KFold+selectIndex) = Acc;
				
				%[acc_ttest,dec_ttest,feature_length_ttest] = classfication_from_ttest(selectedTrainData,selectedTestData,trainLabel{i},testLabel{1},0.05);
				%Kfold_Acc_ttest(lambdaInd,lambda2Ind,(repeat-1)*KFold+selectIndex) = acc_ttest(1);
				%Kfold_length_ttest(lambdaInd,lambda2Ind,(repeat-1)*KFold+selectIndex) = feature_length_ttest;
				clear decision_values testIndex trainIndex testLabel trainLabel testData trainData predictLabel  predictScore 
         
            end   
         end
		lambda2Ind = lambda2Ind + 1;
	end
	lambdaInd = lambdaInd + 1;
end

lambdaInd = 1;
for lambda = lambdas
	lambda2Ind = 1;
	for lambda2 = lambdas2
		for repeat=1:repeats
			predictScore=all_decision_values{lambdaInd,lambda2Ind,repeat};
			predictLabel=predictScore;
			predictLabel(predictScore>0.5)=1;
			predictLabel(predictScore<=0.5)=-1;
			
			Acc = (sum(predictLabel(labels{1}==1)==1)+sum(predictLabel(labels{1}==-1)==-1))/size(labels{1},1);
			Sen = mean(predictLabel(labels{1}==1)==1);
			Spe = mean(predictLabel(labels{1}==-1)==-1);
			Auc = calAUC(predictScore, labels{1});
			
			Acc_list(repeat)=Acc;
			Sen_list(repeat)=Sen;
			Spe_list(repeat)=Spe;
			Auc_list(repeat)=Auc;
		end
		Kfold_Acc_std=std(Kfold_Acc(lambdaInd,lambda2Ind,:));
        feature_length_mean='';
        feature_length_str='';
        for k=1:taskNums
            feature_length_str=strcat(feature_length_str,num2str(feature_lenght(k)),'_');
			feature_length_mean=strcat(feature_length_mean,num2str(feat_select_length{lambdaInd,lambda2Ind,k}),'_');
        end
        
        fprintf(fid, '****************************************\n');
		fprintf(fid, 'all******* lambda = %f, lambda2 = %f *********\n', lambda, lambda2);
		fprintf(fid, '* feature_length =%s, after selcet Mean length=%s *\n',feature_length_str,feature_length_mean)
		fprintf(fid, ' Mean classification accuracy std: %0.4f%% \n', Kfold_Acc_std);
		fprintf(fid, ' Mean classification accuracy: %0.2f%% \n', 100*mean(Acc_list));
		fprintf(fid, ' Mean classification sensitivity: %0.2f%% \n', 100*mean(Sen_list));
		fprintf(fid, ' Mean classification specificity: %0.2f%% \n', 100*mean(Spe_list));
		fprintf(fid, ' Mean classification auc: %0.4f \n', mean(Auc_list));
		fprintf(fid, '****************************************\n');
        
		if maxAcc < mean(Acc_list)
			maxAcc = mean(Acc_list);
			maxSen = mean(Sen_list);
			maxSpe = mean(Spe_list);
			maxAuc = mean(Auc_list);
			lambda_best=lambda;
			lambda2_best=lambda2;
            feature_length_best=feature_length_mean
		end
		lambda2Ind = lambda2Ind + 1;
	end
	lambdaInd = lambdaInd + 1;
end

fprintf(fid, '*****************Optimal result****************\n');
fprintf(fid, '******* lambda_best = %f, lambda2_best = %f *********\n', lambda_best, lambda2_best);
fprintf(fid, ' selcet feature Mean length: %s \n', feature_length_best);
fprintf(fid, ' Mean classification accuracy: %0.2f%% \n', 100*maxAcc);
fprintf(fid, ' Mean classification sensitivity: %0.2f%% \n', 100*maxSen);
fprintf(fid, ' Mean classification specificity: %0.2f%% \n', 100*maxSpe);
fprintf(fid, ' Mean classification auc: %0.4f \n', maxAuc);
fprintf(fid, '****************************************\n');



save_name=strcat('UM_Inter_bn_246_',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'all_decision_values', '-mat' )

save_name=strcat('UM_Inter_bn_246_all_pre_label',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'all_pre_label', '-mat' )

save_name=strcat('UM_Inter_bn_246_featselectindex',char(throld),'_',num2str(repeats),'.mat')
save_path=fullfile(save_folder,save_name);
save(save_path, 'featselectindex', '-mat' )

time = toc;
fprintf(fid, 'total running times = %0.2f\n', time);
fprintf('total running times = %0.2f\n', time);
fclose(fid);
fprintf('vote_Lasso End\n');

