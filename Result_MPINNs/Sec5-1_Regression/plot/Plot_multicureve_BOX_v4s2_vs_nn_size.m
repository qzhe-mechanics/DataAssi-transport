%% Error Plot v.s. NN configuration
%
% (C) Qizhi "Kai-Chi" He (qizhi.he@pnnl.gov), PNNL

% **********************************************************************

close all
clear
format long

type_data = 43;
name_test = ['CADn_v6s3_ds',num2str(type_data),'_Error'];

type_pro  = 1;   % 1: plot k, 2: plot h; 3: C; 11: plot K but no x-axis
type_xlab = 600;  % The x-axis...; 11: prescribed x-label; 600: Neural parameter

type_plot = 40;
if_log    = 0; % 1: logy; 2: log x and y

N_s = 10; % Number of NNs initialization

if type_data ==41
    foldername = './ds41_pro31_opt21m1k400k_lr2e4_s10/';    
    mainFolder = {  'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr1e3_mea2_Nk20k_nn631',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn632',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn633',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn634',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn635',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn636',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn637',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn638',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn639',...
                    'CADn_v6s3_ds41_pro31_opt21m1ki400k_lr2e4_mea2_Nk20k_nn630'};
elseif type_data ==42
    foldername = './ds42_pro31_opt21m1k300k_lr2e4_s10/';
    mainFolder = {  'CADn_v6s3_ds42_pro31_opt21m1ki300k_lr1e3_mea2_nn631_mac',...
                    'CADn_v6s3_ds42_pro31_opt21m1ki300k_lr2e4_mea2_nn632_mac',...
                    'CADn_v6s3_ds42_pro31_opt21m1ki300k_lr2e4_mea2_nn633_mac',...
                    'CADn_v6s3_ds42_pro31_opt21m1ki300k_lr2e4_mea2_nn634_mac',...
                    'CADn_v6s3_ds42_pro31_opt21m1ki300k_lr2e4_mea2_nn635_mac',...
                    'CADn_v6s3_ds42_pro31_opt21m1ki300k_lr2e4_mea2_nn636_mac',...
                    'CADn_v6s3_ds42_pro31_opt21m1ki300k_lr2e4_mea2_nn637_mac',...
                    'CADn_v6s3_ds42_pro31_opt21m1ki300k_lr2e4_mea2_nn638_mac'};
elseif type_data ==43
    foldername = './ds43_pro31_opt21m1k200k_lr2e4_s10/';
    mainFolder = {  'CADn_v6s3_ds43_pro31_ns10_opt21m1ki200k_lr2e4_mea2_nn631_mac',...
                    'CADn_v6s3_ds43_pro31_ns10_opt21m1ki200k_lr2e4_mea2_nn632_mac',...
                    'CADn_v6s3_ds43_pro31_ns10_opt21m1ki200k_lr2e4_mea2_nn633_mac',...
                    'CADn_v6s3_ds43_pro31_ns10_opt21m1ki200k_lr2e4_mea2_nn634_mac',...
                     'CADn_v6s3_ds43_pro31_ns10_opt21m1ki200k_lr2e4_mea2_nn635_mac',...
                     'CADn_v6s3_ds43_pro31_ns10_opt21m1ki200k_lr2e4_mea2_nn636_mac',...
                     'CADn_v6s3_ds43_pro31_ns10_opt21m1ki200k_lr2e4_mea2_nn637_mac',...
                     'CADn_v6s3_ds43_pro31_ns10_opt21m1ki200k_lr2e4_mea2_nn638_mac'};
end
num_dataset = numel (mainFolder);

if type_xlab == 600
    x = [20000];
    x_nn_dof = [261 921 1981 3441 5301	7561 10221 13281 16741 20601]; % Measurement point
    nn_label_all = {'10','20','30','40','50','60','70','80','90','100'};
    nn_label = cell(1,num_dataset);
    for i=1:num_dataset
        nn_label{i} = nn_label_all{i};
    end
    x_nn_label_all = [10:10:100];
    x_nn_label = x_nn_label_all(1:num_dataset);
end


marker = {'-ok','-*r','-xb','-^m','->'};
marker_err = {'--k','--r','--b','--m','--'};

%% Data preparation
box_XM  = zeros(N_s,num_dataset);   %  all data
recor_XM = zeros(3,num_dataset);    %  mean, variance, and mean time.

for i_curve = 1: num_dataset
    subFolder_name = mainFolder{i_curve};
    data_eg1 = load([foldername,subFolder_name,'/','record_data','.out']);
    
    [n,m] = size(data_eg1); num_test = n/N_s;
    
    k_mean = zeros(num_test,m);
    k_std  = zeros(num_test,m);
    
    for i = 1:length(x)
        data_i = data_eg1((i-1)*N_s+1:i*N_s,:);
        k_mean(i,:) = mean(data_i,1);
        k_std(i,:)  = std(data_i,0,1); % 0: means use N-1 for normalization; 1: means along colomn.
    end
    
    if type_pro == 1
        idex_i = 2;
    elseif type_pro == 2
        idex_i = 3;
    elseif type_pro == 3
        idex_i = 4;
    end
    recor_XM(1,i_curve) = k_mean(i,idex_i);
    recor_XM(2,i_curve) = k_std(i,idex_i);
    recor_XM(3,i_curve) = k_mean(i,1);
    box_XM(:,i_curve) = data_i(:,idex_i);

end

%% Plot Start
h1=figure(1);
% grid on;
box on;
if type_plot == 4 % use box plot
    b = boxplot(box_XM,'Labels',nn_label,'Whisker',1.5,'OutlierSize',10,'Widths',0.5);
    set(b,{'linew'},{1.2})
elseif type_plot == 41 % use box and both
    b = boxplot(box_XM,'Labels',nn_label,'Whisker',1.5,'OutlierSize',10,'Widths',0.5);
    set(b,{'linew'},{1.2})
    hold on
    errorbar([1:num_dataset]+0.3,recor_XM(1,:), recor_XM(2,:), marker{1},'linewidth',1.5,'MarkerSize',5,'CapSize', 8);
elseif type_plot == 40  % use box and both
    errorbar([1:num_dataset],recor_XM(1,:), recor_XM(2,:), marker{1},'linewidth',1.5,'MarkerSize',5,'CapSize', 8);
end
% ** Plan size
set (gcf,'Position',[0 0 800 600]);
set	(gca, 'fontsize',20,'FontName','Times New Roman',...
    'LabelFontSizeMultiplier',1.0);

% ** xlable and ylabel
if type_plot == 4 || type_plot == 41 || type_plot == 40
    xlabel('Number of neurons in each hidden layer $m_h$','FontSize',28,'Interpreter','latex')
    ylabel('Relative error $\epsilon$','FontSize',28,'Interpreter','latex')
    if type_data == 41
        ylim([0,0.05])
        set(gca,'ytick',[0:0.01:0.05]) % yticks([0:0.005,0.02])        
    elseif type_data == 42 
        ylim([0,0.02])
        set(gca,'ytick',[0:0.005:0.02]) % yticks([0:0.005,0.02])
    elseif type_data == 43
        ylim([0,0.01])
        set(gca,'ytick',[0:0.002:0.01]) % yticks([0:0.005,0.02])
    end
    
   if type_plot == 40
     xlim([0,num_dataset+0.5])
     num_xlable = num_dataset;
     set(gca,'xtick',[1:num_xlable],'xticklabel',nn_label)
   end
end
hold off

figure(h1)
if type_pro == 1
    lstr_name = ['PlotCorr_k_',name_test];
elseif type_pro == 2
    lstr_name = ['PlotCorr_h_',name_test];
elseif type_pro == 3
    lstr_name = ['PlotCorr_c_',name_test];
end
% 1: .fig; 2: .eps; 3: .pdf; 4: -dtiff 300dpi ; 5: -dtiff 500dpi ; 6 -dpng
% 300dpi; 7 -dpng -r500
type_fig_save = [0 1 0 0 0 0 1];
sub_fig_print_v2(lstr_name,type_fig_save)

