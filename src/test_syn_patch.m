% This file produces the result in the synthetic image patch outlier
% detection example in Section 4.2


clear all; rng(06102017);

%% image of stripes
L=100;
[xx,yy]=meshgrid(-L+.5:L,-L+.5:L);
xx=xx/L;
yy=yy/L;


rho=.05;
im=(1+cos(((xx*rho+yy+1.5).^2*2)*2*pi))/2;
[L1,L2]=size(im);

% add the outlier
sig_outlier = 0.05;
im2= exp(-(xx.^2+yy.^2)/(2*sig_outlier^2));

im2 = im2*0.6;

im=im+im2;

figure(1),clf;
imagesc(im); axis off;
colormap(gray);colorbar();
set(gca,'FontSize',20);


%% patch of image
patchDim=9;
stride=3; 
[X, topleftOrigin] = im2patch2(im,stride,patchDim);

X=X';
[n,dim]=size(X);

xz=zeros(n,1);
yz=zeros(n,1);
for i=1:n
    xz(i)=xx(topleftOrigin(i,2),topleftOrigin(i,1));
    yz(i)=yy(topleftOrigin(i,2),topleftOrigin(i,1));
end

lz=sqrt(n);

Z=(xz*rho+yz+1.5).^2; %the freq of background modulation
% figure(2),imagesc(reshape(Z,lz,lz));colorbar();
% title('Z: freq of background modulation')
% drawnow();

n

%% ground truth labels, 1: outlier, 0:background
delta_outlier = 0.01;

im2val = zeros(n,1);
for i=1:n
    im2val(i) = im2( topleftOrigin(i,2)+floor(patchDim/2),topleftOrigin(i,1)+floor(patchDim/2));
end

tau2 = quantile( im2val,  1-delta_outlier);
labels = double( im2val > tau2);

%% no subsample

data = X;
ldata = labels;
zdata= Z;
c = topleftOrigin;


figure(2),clf;
imagesc(reshape(labels,lz,lz));
title('outlier label'); axis off
set(gca,'FontSize',20);
colormap(gray)

%% construct the graph by ZP-spec (self-tune)
dis=pdist(X);

k_selftune = 32; 
D_sort = sort(squareform(dis),2);
sigma_ZP = D_sort(:, k_selftune);
minsig=min(sigma_ZP)

W=exp(-(squareform(dis).^2)./(sigma_ZP*sigma_ZP')/2);
W=W-diag(diag(W));

% degree of nodes
dW = sum(W,2);


%% eig of graph
tic
[v,d] = eig(W,diag(dW));
toc

d = diag(d);
[lambda,tmp] = sort(d,'descend');
psi = v(:,tmp);


%% parameters
Kmin = 100;
Kmax = 400;
kI_list = (Kmin:1:Kmax)';
ncol = numel(kI_list);
Nmax = kI_list(end);

figure(5),clf;
plot(lambda,'x-');
grid on;
title('lambda')

figure(6),clf;
imagesc( abs(psi(:,1:Nmax)));
title('|psi|')

figure(7),clf;
idxj=[2,3,4];
%scatter3(psi(:,idxj(1)),psi(:,idxj(2)),psi(:,idxj(3)),40,Z,'o','filled');
scatter3(psi(:,idxj(1)),psi(:,idxj(2)),psi(:,idxj(3)),40,labels,'o','filled');
title(num2str(idxj))
grid on; colormap(jet);

figure(8),clf;
%find indicator psi index which maximize variation on the outlier
cmass = sum(abs(psi(labels ==1,1:Nmax)),1);
[~,jsort] = sort(cmass,'descend');
idxj = jsort(1:3);
%scatter3(psi(:,idxj(1)),psi(:,idxj(2)),psi(:,idxj(3)),40,Z,'o','filled');
scatter3(psi(:,idxj(1)),psi(:,idxj(2)),psi(:,idxj(3)),40,labels ,'o','filled');
title(num2str(idxj))
grid on; colormap(jet);

%% embedding norm
kI = 250; %150, 200, 300, 350
sI = sum(psi(:,1:kI).^2, 2);

figure(11),clf;
imagesc(reshape(sI,lz,lz));
title(sprintf('|I|=%d',kI))
axis off; 
set(gca,'FontSize',20);
colormap(gray);

%% plot of eigenvectors

jlist = [2,3,5,10,20,50,100,139,232,277];
figure(17); clf;
for i=1:numel(jlist)
    j = jlist(i);
    subplot(2,5,i);
    imagesc(reshape(psi(:,j),lz,lz));
    title(sprintf('k=%d',j))
    axis off;
    set(gca,'FontSize',20);
end
colormap(gray);


%% F1 score
f1s = zeros(ncol,1);

for icol=1:ncol
    
    kI = kI_list(icol);
    
    %% sI
    sI = sum(psi(:,1:kI).^2, 2);
    
    % confusion matrix and f score
    s_th = quantile( sI, 1-delta_outlier);
    
    label_true = (labels > 0);
    label_pred = (sI > s_th);
    M = confusionmat( label_true, label_pred);
    
    prec = M(2,2)/(M(2,2)+M(1,2));  %tp/(tp+fp)
    recl = M(2,2)/(M(2,2)+M(2,1) ); %tp/(tp+fn)
    if ~(prec+recl > 0)
        F1=0;
    else
        F1 = 2*prec*recl/(prec+recl);
    end
    
    f1s(icol) = F1;
    
end


figure(21),clf;
plot(kI_list, f1s,'x-');
grid on;
set(gca,'FontSize',20);
title('F1 score');
 xlabel('|I|');


return;

%%
% the rest of the code implements random subsample of the patches and
% compute the mean and std of the F1 score. It takes ~10 min to finish.

ifplot =1;

%% replica of the experiment by randomly subsample the image patches
nrun = 100; 

nn = 3000;

f1s = zeros(ncol,nrun);
for  irun=1:nrun
    %% random subsample
    
    idxnn = randperm( n, nn);
    idxnn = sort( idxnn, 'ascend');
    
    data = X(idxnn, :);
    ldata = labels(idxnn); 
    
    zdata= Z(idxnn, :);
    
    c = topleftOrigin(idxnn,:);
    
    figure(2),clf;
    scatter(c(:,1), c(:,2), 40, ldata, 's','filled');
    title('patch and outlier label')
    %colormap(gray);
    
    %% construct the graph by ZP-spec (self-tune)
    dis=pdist(data);
    D_sort = sort(squareform(dis),2);
    sigma_ZP = D_sort(:, k_selftune);
    
    
    W=exp(-(squareform(dis).^2)./(sigma_ZP*sigma_ZP')/2);
    W=W-diag(diag(W));
    
    % degree of nodes
    dW = sum(W,2);
    
    %% eig of graph
    tic
    [v,d] = eig(W,diag(dW));
    toc
    
    d = diag(d);
    [lambda,tmp] = sort(d,'descend');
    psi = v(:,tmp);
    
    Nmax = kI_list(end);
    
    
    %% vis the spectrum
    if ifplot
        figure(5),clf;
        plot(lambda,'x-');
        grid on;
        title('lambda')
        
        figure(6),clf;
        imagesc( abs(psi(:,1:Nmax)));
        title('|psi|')
        
        figure(7),clf;
        idxj=[2,3,4];
        scatter3(psi(:,idxj(1)),psi(:,idxj(2)),psi(:,idxj(3)),40,ldata,'o','filled');
        title(num2str(idxj))
        grid on; colormap(jet);
        
        figure(8),clf;
        cmass = sum(abs(psi(ldata ==1,1:Nmax)),1);
        [~,jsort] = sort(cmass,'descend');
        idxj = jsort(1:3);
        scatter3(psi(:,idxj(1)),psi(:,idxj(2)),psi(:,idxj(3)),40,ldata,'o','filled');
        title(num2str(idxj))
        grid on; colormap(jet);
    end
    
    %% embedding norm
    kI = 250;
    
    sI = sum(psi(:,1:kI).^2, 2);
    
    if ifplot
        figure(11),clf;
        scatter(c(:,1), c(:,2), 40, sI, 's','filled');
        title(sprintf('S, I=%d',kI))
        colormap(gray);
    end

    
    %% F1 score
   
    for icol=1:ncol
        
        kI = kI_list(icol);
        
        %% sI
        sI = sum(psi(:,1:kI).^2, 2);
        
        % confusion matrix and f score
        s_th = quantile( sI, 1-delta_outlier);
        
        label_true = (ldata > 0);
        label_pred = (sI > s_th);
        M = confusionmat( label_true, label_pred);
        
        prec = M(2,2)/(M(2,2)+M(1,2));  %tp/(tp+fp)
        recl = M(2,2)/(M(2,2)+M(2,1) ); %tp/(tp+fn)
        if ~(prec+recl > 0)
            F1=0;
        else
            F1 = 2*prec*recl/(prec+recl);
        end
        
        
        f1s(icol,irun) = F1;
        
    end
    
    if ifplot
        figure(21),clf;
        plot(kI_list, f1s(:,irun),'x-');
        grid on;
        set(gca,'FontSize',20);
        title('F1 score');
        xlabel('|I|');
        
    end
end

%%
figure(22),clf; hold on;
f =f1s;
mf = mean(f,2);
stdf = std(f,1,2);
errorbar(kI_list, mf, stdf,'.-')
[~, jmax] = max(mf);
scatter( kI_list(jmax), mf(jmax),'xr');
grid on; axis([kI_list(1), kI_list(end),0,1])
title(sprintf('k_{ST}=%d, |I|=%d, f1=%4.2f (%4.2f)',k_selftune, kI_list(jmax),...
                                mf(jmax)*100,stdf(jmax)*100));
set(gca,'FontSize',20);


