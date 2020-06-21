function plotsigmalog(Sigma, r)
    sd = sum(diag(Sigma));
    m = diag(Sigma);
    n = length(m);
    
    semilogy(1:r, m(1:r)/sd,'r*','Linewidth', [3])
    hold on 
    semilogy(r+1:n, m(r+1:end)/sd,'k*','Linewidth', [3])
    grid on
    legend('active modes', 'lost modes')
    ylabel('Singular value')
    xlabel('k')
    title('Singular values');
    
end