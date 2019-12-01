function sub_fig_print_v2(lstr_name,type_fig_save)
%% Create @ 2018.08.25
%% Update parameter @ 2019.05
%% Update parameter @ 2019.08
% 1: .fig; 2: .eps; 3: .pdf; 4: -dtiff 300dpi ; 5: -dtiff 500dpi ; 6 -dpng
% 300dpi; 7 -dpng -r500
s = length(type_fig_save);
for i = 1: s
    if type_fig_save(i) == 1
        switch i
            case 1
                savefig(lstr_name)
            case 2
                print(gcf,'-depsc','-r500',[lstr_name,'.eps'])
            case 3
                %                 print(gcf,'-dpdf','-fillpage',[lstr_name,'.pdf'])
                print(gcf,'-dpdf','-bestfit',[lstr_name,'.pdf'])
                %                 print('FillPageFigure','-dpdf','-fillpage')       
            case 4
                print(gcf,'-dtiff','-r300',[lstr_name,'_Low','.tiff'])
            case 5
                print(gcf,'-dtiff','-r500',[lstr_name,'.tiff'])            
            case 6
%                 print(gcf,'-dpng','-r300',['Low_',lstr_name,'.png'])
                print([lstr_name,'_Low'],'-dpng','-r300');
            case 7
%                 print(gcf,'-dpng','-r300',['Low_',lstr_name,'.png'])
                print(lstr_name,'-dpng','-r500');
        end
    end
end
%                 saveas(gcf,[lstr_name,'.jpg'])
end