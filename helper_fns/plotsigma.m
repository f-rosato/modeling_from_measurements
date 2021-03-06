function plotsigma(Sigma, r)
    sd = sum(diag(Sigma));
    m = diag(Sigma);
    n = length(m);
    
    plot(1:r, m(1:r)/sd,'ro','Linewidth', [3])
    hold on 
    plot(r+1:n, m(r+1:end)/sd,'ko','Linewidth', [3])
    grid on
    legend('active modes', 'lost modes')
    ylabel('Singular value')
    xlabel('k')
    title('Singular values');
    
end