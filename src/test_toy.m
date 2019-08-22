% This file produces the result in the toy data example in Section 4.1
%
% rule to choose |I|: |I| ~ K*(1-delta)/delta, thus by setting K ~ delta,
% we exepct |I| in about the same range over vaying values of K and delta


clear all; rng(2018);

%% parameter of data 

dim =2;


% example 2

%K = 10;
%delta = 0.1;


% example 1
K = 2;
delta = 0.02;


%
n = 5000;

n11 = floor(n*delta/K); %size of each sub-cluster in C
n1 = n11*K; %size of C

dr1 = 0.1;
r1 = 1+dr1;
eps1 = 0.02;

n2 = n-n1; %size of B
eps2 = 0.01;
h3 = 0.5;

%%

kself = 8; %16 %32
kI = 23; %36; %60 %70  %K

%% sample data
rx1 = r1*ones(K,1);

ifrndx1 = 0;

if ifrndx1
    x1 = randn(K,dim); x1 = x1./sqrt(sum(x1.^2,2));
else
    t1 = ((1:1:K)/K)';
    t1 = t1 + (rand(K,1)-0.5)*(1/K);
    t1 = mod(t1,1);
    x1 = [cos(t1*2*pi),sin(t1*2*pi)];
end

x1 = diag(rx1)*x1;
x1 = kron( x1, ones(n11,1));
x1 = x1 + randn(n1,dim)*eps1;

x2 =  randn(n2,dim); x2 = x2./sqrt(sum(x2.^2,2));
x2 = x2 + randn( n2, dim)*eps2;
[~,tmp] = sort(x2(:,1),'ascend');
x2 = x2(tmp,:);

data = cat(1, x1, x2);
labels = cat(1, kron( (1:K)', ones(n11,1)), zeros(n2,1)); %1 for outlier

%% construct W
k_nn        = 10*kself;
[nnInds, nnDist] = knnsearch(data,data,'k', k_nn);
sigmaKvec = nnDist(:,kself);

rowInds = kron((1:n)', ones(k_nn,1));
colInds = reshape(nnInds', k_nn*n, 1);
vals    = reshape(nnDist', k_nn*n, 1);
autotuneVals = sigmaKvec(rowInds).*sigmaKvec(colInds);
Kvals = exp( -vals.^2 ./ (2*autotuneVals)  );
W = sparse(rowInds, colInds, Kvals, n, n);
W = (W + W')/2;

%% vis
figure(1),clf;
scatter( data(:,1), data(:,2), 40, double(labels >0),'.');
grid on; colormap(jet); %colorbar();
title('data');set(gca,'FontSize',20);
axis([-1.5, 1.5, -1.5, 1.5])

figure(2),clf;
spy(W)
title(sprintf('W, ST=%d',kself));
set(gca,'FontSize',20);

%% eig
Nmax = 100;
tic,
[v,d] = eigs(W, diag(sum(W,2)), Nmax, 'la');
toc
d= diag(d);
[lambda,tmp] = sort(d,'descend');
psi = v(:,tmp);

%%
figure(4),clf;
plot(lambda,'x-','Linewidth',2);
grid on; title('\lambda_k');
set(gca,'FontSize',20);
xlabel('k'); axis([0,100,0.88 1.005])

figure(5),clf;
imagesc(abs(psi))
title('|Psi|')

figure(6),clf;
%for j=1:16
j_list = [2,3,4,5,8,9,10,11]; %K=2
%j_list = [2,4,6,8,12,14,16,18]; %K=10
for j=1:8
    %subplot(4,4,j), hold on;
    subplot(2,4,j), hold on;
    jid = j_list(j);
    scatter( data(:,1), data(:,2), 40, abs(psi(:,jid)),'.');
    grid on; colormap(jet); colorbar();
    title(sprintf('k=%d',jid)); set(gca,'FontSize',20);
    axis([-1.5, 1.5, -1.5, 1.5])
end

%%
sI = sum(psi(:,1:kI).^2, 2);

% confusion matrix and f score
s_th = quantile( sI, n2/n);

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
F1


figure(11),clf; hold on;
hist(sI,100);
scatter(s_th,0,100,'xr')
grid on;set(gca,'FontSize',20);
title('Histogram of S')

figure(12),clf;
plot(sI); grid on;
title('S');set(gca,'FontSize',20);

%%
figure(13),clf;
scatter( data(:,1), data(:,2), 40, sI,'.');
grid on; colormap(jet); colorbar();
title(sprintf('S, |I|=%d', kI));
set(gca,'FontSize',20);
axis([-1.5, 1.5, -1.5, 1.5])


return;

%%
% the rest of the code implements random replicas of the experiment and
% computes the mean and std of the F1 score. It takes ~20 min to finish.

%% 

nrun =100;

kself_list = (2.^(2:1:7))';

Kmin = 2;
Kmax = floor(K/delta);
kI_list = (Kmin:1:Kmax)';

nrow = numel(kself_list);
ncol = numel(kI_list);

f1s=zeros( nrow, ncol, nrun);


for irun=1:nrun
    
    fprintf('-- %d --\n',irun)
    
    
    %% sample data
    rx1 = r1*ones(K,1);
    
    if ifrndx1
        x1 = randn(K,dim); x1 = x1./sqrt(sum(x1.^2,2));
    else
        t1 = ((1:1:K)/K)';
        t1 = t1 + (rand(K,1)-0.5)*(1/K);
        t1 = mod(t1,1);
        x1 = [cos(t1*2*pi),sin(t1*2*pi)];
    end

    x1 = diag(rx1)*x1;
    x1 = kron( x1, ones(n11,1));
    x1 = x1 + randn(n1,dim)*eps1;
    
    x2 =  randn(n2,dim); x2 = x2./sqrt(sum(x2.^2,2));
    x2 = x2 + randn( n2, dim)*eps2;
    [~,tmp] = sort(x2(:,1),'ascend');
    x2 = x2(tmp,:);
    
    data = cat(1, x1, x2);
    labels = cat(1, kron( (1:K)', ones(n11,1)), zeros(n2,1));
    
    %%
    k_nn_max        = min(10*kself_list(end),n);
    [nnInds_max, nnDist_max] = knnsearch(data,data,'k', k_nn_max);
    
    for irow=1:nrow
        
        kself = kself_list(irow);
        
        %% construct W
        k_nn        = min(10*kself,n);
        
        nnInds = nnInds_max(:,1:k_nn);
        nnDist = nnDist_max(:,1:k_nn);
        
        sigmaKvec = nnDist(:,kself);
        
        rowInds = kron((1:n)', ones(k_nn,1));
        colInds = reshape(nnInds', k_nn*n, 1);
        vals    = reshape(nnDist', k_nn*n, 1);
        autotuneVals = sigmaKvec(rowInds).*sigmaKvec(colInds);
        Kvals = exp( -vals.^2 ./ (2*autotuneVals)  );
        W = sparse(rowInds, colInds, Kvals, n, n);
        W = (W + W')/2;
        
        
        %% eig
        Nmax = kI_list(end);
        tic,
        [v,d] = eigs(W, diag(sum(W,2)), Nmax, 'la');
        toc
        d= diag(d);
        [lambda,tmp] = sort(d,'descend');
        psi = v(:,tmp);
        
        %%
        for icol=1:ncol
            
            kI = kI_list(icol);
            
            %% sI
            sI = sum(psi(:,1:kI).^2, 2);
            
            % confusion matrix and f score
            s_th = quantile( sI, n2/n);
            
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
            
            f1s(irow,icol,irun) = F1;
            
        end
        
        figure(99),clf; hold on;
        plot(kI_list, reshape(f1s(irow,:,irun), [ncol,1])', 'x-' );
        grid on;
        xlabel('|I|');
        title(sprintf('kself=%d',kself))
        
    end
    
    
end
    
%%
mf1 = mean(f1s, 3);

figure(21),clf;

for irow =1:4
    kself=kself_list(irow);
    subplot(2,2,irow), hold on;
    
    f = reshape(f1s(irow, :, :), [ncol, nrun]) ;
    mf = mean(f,2);
    stdf = std(f,1,2);
    errorbar(kI_list, mf, stdf,'.-')
    
    [~, jmax] = max(mf);
    scatter( kI_list(jmax), mf(jmax),'xr');
    
    
    grid on; axis([1, kI_list(end),0,1])
    xlabel('|I|'); 
    title(sprintf('k_{ST}=%d, |I|=%d, f1=%4.2f (%4.2f)',kself, kI_list(jmax),...
                                mf(jmax)*100,stdf(jmax)*100));
    set(gca,'FontSize',20);
end


