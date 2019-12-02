%% Plot Error Plot for Measurements with different Colloca
% version 1
% (C) Qizhi "Kai-Chi" He (qizhi.he@pnnl.gov), PNNL

%**********************************************************************
% ---------------------------------------------------------------------
% History: <mm/dd/yyyy> [key] {Description} [note]
% <05,29,2019> [verision 1]
% <08,11,2019> [verision 2]
% <08,27,2019> [verision 3] {Select mean or error}
% <09,09,2019> [verision 4] {Select log or not log}

% Read other folder
% ---------------------------------------------
% **********************************************************************

close all
clear
format long
% folder_dir = '/Users/heqi220/Dropbox/PNNL/Proj_UQ/Group_Code/PINN-CAD-master-v6s2/Output_bm/temp_ds51_ts3/'


type_pro  = 3; % 1: plot k, 2: plot h; 3: C
if type_pro == 3
    %     C
    mainFolder = {
        'CADn_v6s1_pro1_ts1s5_n23_opt1_c64',...
        'CADn_v6s1_pro1_ts1s5_n23_opt1_c128',...
        'CADn_v6s1_pro1_ts1s5_n33_opt1_c64',...
        'CADn_v6s1_pro1_ts1s5_n33_opt1_c128',...
        'CADn_v6s1_pro1_ts1s5_n43_opt1_c64',...
        'CADn_v6s1_pro1_ts1s5_n43_opt1_c128'};
else
    % K & h
    mainFolder = {  'CADn_v6s1_pro2_ts3s5_n23n13_opt1_f200',...
        'CADn_v6s1_pro1_ts1s5_n23_opt1_c64',...
        'CADn_v6s1_pro1_ts1s5_n23_opt1_c128',...
        'CADn_v6s1_pro1_ts1s5_n33_opt1_c64',...
        'CADn_v6s1_pro1_ts1s5_n33_opt1_c128',...
        'CADn_v6s1_pro1_ts1s5_n43_opt1_c64',...
        'CADn_v6s1_pro1_ts1s5_n43_opt1_c128'};
end

name_test = 'CADn_v6s1_pro1_ts4s5_nn02';

type_xlab = 12;  % The x-axis...
type_plot = 1; % 1: only mean; 2: error plot
if_log  = 0;

num_dataset = numel (mainFolder);
N_s = 5; % Number of NNs initialization
x = [16	36	48	64	80	96]; % Measurement point

% marker = {'-ob','--xb','-or','--xr','-ok','-+k'};
% marker_err = {'-b','--b','-r','--r','-k','--k'};

% marker = {'-xb','->','--.r'}; % Three curve 128

if type_pro == 3
    marker = {'-*r','-xb','-^m','->','--.k','--.r'};
else
    marker = {'-ok','-*r','-xb','-^m','->','--.k','--.r'};
end

% marker_err = {'--k','--r','--b','--m','--'};
% lstr_leng = {'hidden layers $[-32-32-]$', 'hidden layers $[-32-32-32-]$', 'hidden layers $[-64-64-]$', 'hidden layers $[-64-64-64-]$', '$k: N_f = 200$'};
% lstr_leng = {'hidden layers $[-16-16-]$', 'hidden layers $[-16-16-16-]$', 'hidden layers $[-32-32-]$', 'hidden layers $[-32-32-32-]$'};



if type_pro == 3
    lstr_leng = {'$C: N_C = 64$ MPINN (shallow)', '$C: N_C = 128$ MPINN (shallow)', '$C: N_C = 64$ MPINN (medium)', '$C: N_C = 128$ MPINN (medium)',...
        '$C: N_C = 64$ MPINN (deep)', '$C: N_C = 128$ MPINN (deep)'};
elseif type_pro == 2
    lstr_leng = {'$h: N_C = 0$ PINN-Darcy', '$h: N_C = 64$ MPINN (shallow)', '$h: N_C = 128$ MPINN (shallow)', '$h: N_C = 64$ MPINN (medium)', '$h: N_C = 128$ MPINN (medium)',...
        '$h: N_C = 64$ MPINN (deep)', '$h: N_C = 128$ MPINN (deep)'};
elseif type_pro == 1
    lstr_leng = {'$K: N_C = 0$ PINN-Darcy', '$K: N_C = 64$ MPINN (shallow)', '$K: N_C = 128$ MPINN (shallow)', '$K: N_C = 64$ MPINN (medium)', '$K: N_C = 128$ MPINN (medium)',...
        '$K: N_C = 64$ MPINN (deep)', '$K: N_C = 128$ MPINN (deep)'};
end

h1=figure(1);

for i_curve = 1: num_dataset
    subFolder_name = mainFolder{i_curve};
    data_eg1 = load([subFolder_name,'/','record_data','.out']);
    
    [n,m] = size(data_eg1); num_test = n/N_s;
    
    if num_test ~= length(x)
        error('Wrong input')
    end
    
    k_mean = zeros(num_test,m);
    k_std  = zeros(num_test,m);
    
    for i = 1:length(x)
        data_i = data_eg1((i-1)*N_s+1:i*N_s,:);
        k_mean(i,:) = mean(data_i,1);
        k_std(i,:)  = std(data_i,0,1); % 0: means use N-1 for normalization; 1: means along colomn.
    end
    
    %     if type_pro == 1
    %         semilogy(x, k_mean(:,2),marker{i_curve},'linewidth',1.5,'MarkerSize',6)
    %     elseif type_pro == 2
    %         semilogy(x, k_mean(:,3),marker{i_curve},'linewidth',1.5,'MarkerSize',6)
    %     elseif type_pro == 3
    %         semilogy(x, k_mean(:,4),marker{i_curve},'linewidth',1.5,'MarkerSize',6)
    %     end
    if type_pro == 1
        idex_i = 2;
    elseif type_pro == 2
        idex_i = 3;
    elseif type_pro == 3
        idex_i = 4;
    end
    
    if type_plot == 1
        if if_log == 0
            plot(x, k_mean(:,idex_i), marker{i_curve},'linewidth',1.5,'MarkerSize',5)
        elseif if_log == 1
            semilogy(x, k_mean(:,idex_i),marker{i_curve},'linewidth',1.5,'MarkerSize',5)
        end
    elseif type_plot == 2
        errorbar(x, k_mean(:,idex_i), k_std(:,idex_i), marker{i_curve},'linewidth',1.5,'MarkerSize',5,'CapSize', 8);           % Plot Error Bars
    end
    hold on
end

% [lgd,objh] = legend(lstr_leng,...
%     'location', 'NE', 'Fontsize', 28,'FontName','Times New Roman');
[lgd,objh] = legend(lstr_leng,...
    'location', 'NE', 'Fontsize', 24,'Interpreter','latex');
objhl = findobj(objh, 'type', 'patch');  %// objects of legend of type patch
set(objhl,'MarkerSize', 10);            %// set marker size as desired
set(lgd,'Box','off');
% set(objhl,'Interpreter','latex');
% annotation('textbox',lgd.Position,'String',{'Some info'},...
%     'VerticalAlignment','Top','Edgecolor','none','Fontsize', 28);

set (gcf,'Position',[0 0 800 600]);
set	(gca, 'fontsize',20,'FontName','Times New Roman',...
    'LabelFontSizeMultiplier',1.0);

if if_log == 1
    ylim([0.001,5]);
elseif if_log == 0
    if type_pro == 1
         ylim([0,0.4]);
    elseif type_pro == 2
        ylim([0,0.04]);
    elseif type_pro == 3
%         ylim([0,0.06]);
        ylim([0,0.2]);
    end
end

if type_pro == 1
    ylabel('Relative error $\epsilon^{K}$','FontSize',28,'Interpreter','latex')
elseif type_pro == 2
    ylabel('Relative error $\epsilon^{h}$','FontSize',28,'Interpreter','latex')
elseif type_pro == 3
    ylabel('Relative error $\epsilon^{C}$','FontSize',28,'Interpreter','latex')
end

if type_xlab == 1
    xlabel('Measurement number $N_K$','FontSize',28,'Interpreter','latex')
elseif type_xlab == 2
    xlabel('Measurement number $N_h$','FontSize',28,'Interpreter','latex')
elseif type_xlab == 12
    xlabel('Measurement number $N_K$ and $N_h$','FontSize',28,'Interpreter','latex')
elseif type_xlab == 3
    xlabel(['Collocation points ','$N_f$'],'FontSize',28,'Interpreter','latex')
elseif type_xlab == 4
    xlabel('Measurement number $N_C$','FontSize',28,'Interpreter','latex')
elseif type_xlab == 5
    xlabel(['Collocation points ','$N_{fc}$'],'FontSize',28,'Interpreter','latex')
end

% title('khr: N_h = 80, N_f = 20, N_c = 800, N_{fc} = 200, N_i = 5','Fontsize',20,'FontName','Times New Roman')
% set		(gca,'fontsize',20,'FontName','Times New Roman');
box on; grid on;
hold off


% 1: .fig; 2: .eps; 3: .pdf; 4: -dtiff 300dpi ; 5: -dtiff 500dpi ; 6 -dpng
% 300dpi; 7 -dpng -r500

figure(h1)

if type_pro == 1
    lstr_name = ['mulplot_k_',name_test];
elseif type_pro == 2
    lstr_name = ['mulplot_h_',name_test];
elseif type_pro == 3
    lstr_name = ['mulplot_c_',name_test];
end
type_fig_save = [0 1 0 0 0 0 0];
sub_fig_print_v2(lstr_name,type_fig_save)

