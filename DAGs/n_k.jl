N = 10000
k = 10
function f_p_q(q)
    if q == k
        return factorial(q)
    a = 0
    for m in 0:q
        a = a + factorial(q-m)*f_p_q(q+1) 
    end
end
fucntion n_k(N, k)
    a = factional(N)
    for i in 0:(N-k+1)
        1/(factorial(N-k+1-i)*factorial(i))*(-1)^i*factorial(i+1)/((factorial(k+i))^2)
    end
end