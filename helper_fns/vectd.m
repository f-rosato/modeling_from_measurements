function time_delayed_vecs = vectd(x1, x2, e, n)

    time_delayed_vecs = [];
    for i = 1:n
        time_delayed_vecs = [time_delayed_vecs; x1(i:e+i)];
        time_delayed_vecs = [time_delayed_vecs; x2(i:e+i)];
    end


end