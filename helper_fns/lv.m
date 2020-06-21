function dudt = lv(b, p, r, d, t, u)
    
    % let's call "u" the vector
    x = u(1);
    y = u(2);

    % update
    xp = (b - p*y)*x;
    yp = (r*x - d)*y;

    dudt = [xp; yp];
    
end