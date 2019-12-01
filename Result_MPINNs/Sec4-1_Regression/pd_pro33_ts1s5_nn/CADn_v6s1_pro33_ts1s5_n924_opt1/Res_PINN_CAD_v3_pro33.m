%% Plot Error Plot for Measurements
% version 2
% (C) Qizhi "Kai-Chi" He (qizhi.he@pnnl.gov), PNNL

%**********************************************************************
% ---------------------------------------------------------------------
% History: <mm/dd/yyyy> [key] {Description} [note]
% <04,18,2019> {Create..} [Done]
% <05,17,2019> [verision 1] {make the legend larger} [Done]
% ---------------------------------------------------------------------
% **********************************************************************

close all
clear
format long

data_eg1 = load('record_data.out');

N_s = 5; % Number of NNs initialization

% x = [10	25	40	60	80	100	120]; % Measurement point
x = [16	32	64	128	256	512];

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

% v = size(Rmtx,2)-1;
% tstat = tinv(0.95,v);                                       % For 95% Confidence Intervals
% Rci = bsxfun(@times, [-1 1], Rsem*tstat);

h1=figure(1); hold on
set (gcf,'Position',[0 0 800 600]);
set	(gca, 'fontsize',20,'FontName','Times New Roman',...
    'LabelFontSizeMultiplier',1.0);
plot(x, k_mean(:,2),'-sr','linewidth',1.5,'MarkerSize',5,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6])
he = errorbar(x, k_mean(:,2), k_std(:,2), '-r','linewidth',1.5);           % Plot Error Bars
% lgd = legend(he, 'k (Mean ± SDV)', 'Location','NE');
[lgd,objh] = legend(he,{'K (mean ± SD)'},...
    'location', 'NE', 'Fontsize', 32,'FontName','Times New Roman');
objhl = findobj(objh, 'type', 'patch');  %// objects of legend of type patch
set(objhl,'MarkerSize', 10);            %// set marker size as desired
set(lgd,'Box','off');
ylabel('Relative error','FontSize',28,'FontName','Times New Roman')
ylim([0,1]);
% xlim([0,100]);
xlabel('C measurements','FontSize',28,'FontName','Times New Roman')
% title('khr: N_h = 80, N_f = 20, N_c = 800, N_{fc} = 200, N_i = 5','Fontsize',20,'FontName','Times New Roman')
% set		(gca,'fontsize',20,'FontName','Times New Roman');
box on; grid on;
hold off

h2=figure(2); hold on
set (gcf,'Position',[100 0 800 600]);
set	(gca, 'fontsize',20,'FontName','Times New Roman',...
    'LabelFontSizeMultiplier',1.0);
plot(x, k_mean(:,3),'-sr','linewidth',1.5,'MarkerSize',5,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6])
he = errorbar(x, k_mean(:,3), k_std(:,3), '-r','linewidth',1.5);           % Plot Error Bars
% lgd = legend(he, 'k (Mean ± SDV)', 'Location','NE');
[lgd,objh] = legend(he,{'h (mean ± SD)'},...
    'location', 'NE', 'Fontsize', 32,'FontName','Times New Roman');
objhl = findobj(objh, 'type', 'patch');  %// objects of legend of type patch
set(objhl,'MarkerSize', 10);            %// set marker size as desired
set(lgd,'Box','off');
ylabel('Relative error','FontSize',28,'FontName','Times New Roman')
ylim([0,1]);
% xlim([0,100]);
xlabel('C measurements','FontSize',28,'FontName','Times New Roman')
% title('khr: N_h = 80, N_f = 20, N_c = 800, N_{fc} = 200, N_i = 5','Fontsize',20,'FontName','Times New Roman')
% set		(gca,'fontsize',20,'FontName','Times New Roman');
box on; grid on;
hold off

h3=figure(3); hold on
set (gcf,'Position',[200 0 800 600]);
set	(gca, 'fontsize',20,'FontName','Times New Roman',...
    'LabelFontSizeMultiplier',1.0);
plot(x, k_mean(:,4),'-sr','linewidth',1.5,'MarkerSize',5,'MarkerEdgeColor','red','MarkerFaceColor',[1 .6 .6])
he = errorbar(x, k_mean(:,4), k_std(:,4), '-r','linewidth',1.5);           % Plot Error Bars
% lgd = legend(he, 'k (Mean ± SDV)', 'Location','NE');
[lgd,objh] = legend(he,{'C (mean ± SD)'},...
    'location', 'NE', 'Fontsize', 28,'FontName','Times New Roman');
objhl = findobj(objh, 'type', 'patch');  %// objects of legend of type patch
set(objhl,'MarkerSize', 10);            %// set marker size as desired
set(lgd,'Box','off');
ylabel('Relative error','FontSize',28,'FontName','Times New Roman')
% xlim([0,100]);
ylim([0,1]);
xlabel('C measurements','FontSize',28,'FontName','Times New Roman')
% title('khr: N_h = 80, N_f = 20, N_c = 800, N_{fc} = 200, N_i = 5','Fontsize',20,'FontName','Times New Roman')
% set		(gca,'fontsize',20,'FontName','Times New Roman');
box on; grid on;
hold off

% 1: .fig; 2: .eps; 3: .pdf; 4: -dtiff 300dpi ; 5: -dtiff 500dpi ; 6 -dpng
% 300dpi; 7 -dpng -r500
name_test = 'CADn_v6_pro33_ts1';
% figure(h1)
% lstr_name = ['plot_k_',name_test];
% type_fig_save = [0 1 0 1 0 0];
% sub_fig_print_v1(lstr_name,type_fig_save)
% figure(h2)
% lstr_name = ['plot_h_',name_test];
% type_fig_save = [0 1 0 1 0 0];
% sub_fig_print_v1(lstr_name,type_fig_save)
figure(h3)
lstr_name = ['plot_c_',name_test];
type_fig_save = [0 1 0 1 0 0];
sub_fig_print_v1(lstr_name,type_fig_save)





