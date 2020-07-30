using LinearAlgebra;


function exponential_kernel(x1, x2, gp_params)
    if length(x1) == 1
        sigma = gp_params[:sigma]; scale = gp_params[:scale];
        return (sigma^2)*exp(-((x1-x2)^2)/(2*scale^2))
    elseif length(x1) == 2
        sigma = gp_params[:sigma]; scale_t = gp_params[:scale_t]; scale_f = gp_params[:scale_f];
        return (sigma^2)*exp( -( ((x1[1]-x2[1])^2)/(2*scale_t^2) + ((x1[2]-x2[2])^2)/(2*scale_f^2) ) )    
    end
end

function ou_kernel(x1, x2, kernel_params)
    if length(x1) == 1
        sigma = kernel_params[1]; scale = kernel_params[2];
        return (sigma^2)*exp(-abs(x1-x2)/(2*scale^2))
    elseif length(x1) == 2
        sigma = kernel_params[1]; scale_t = kernel_params[2]; scale_f = kernel_params[2];
        return (sigma^2)*exp( -( abs(x1[1]-x2[1])/(scale_t^2) + abs(x1[2]-x2[2])/(scale_f^2) ) )    
    end
end

function generate_covariance_matrix(ts, gp_params)
    
    #Doesn't iterate through nxn values, uses symmetry
    n=length(ts)
    cov = zeros(n,n)
    #Returns a lower triangular matrix
    sigma = gp_params[:sigma]; epsilon = gp_params[:epsilon]
    for i = 1:n
        for j = 1:i
            cov[i, j] = i == j ? (sigma^2 + epsilon + 1e-4) : exponential_kernel(ts[i], ts[j], gp_params)
        end
    end
    #Need to transform into symmetric:
    return cov + transpose(cov) - Diagonal(cov)
    
end

function generate_cross_covariance_matrix(ts1, ts2, gp_params)
    
    #Doesn't use symmetry
    n1=length(ts1)
    n2=length(ts2)
    cov = zeros(n1,n2)
    for i = 1:n1
        for j = 1:n2
            cov[i, j] = exponential_kernel(ts1[i], ts2[j], gp_params)
        end
    end
    return cov

end

function generate_mean(t, mu)
    if isa(mu, Number)
        return mu*ones(length(t))
    else
        #get_gp_spectrotemporal returns the time points in the format
        #[ (t1,f1), (t1,f2) ... (t1,fn), (t2, f1), ... ]
        #So the mean spectrum (which is for a single timepoint, len=fn)
        #should be repeated as many time points as there are
        #length(t) should be divisible by length(mu) with no remainder
        @assert (length(t) % length(mu) == 0)
        n_timepoints = Int(length(t)/length(mu))
        return repeat(mu, n_timepoints)
    end
end


function get_mu_cov(xs, gp_params)
    cov = generate_covariance_matrix(xs, gp_params)
    mu = generate_mean(xs, gp_params[:mu])
    return mu, cov 
end


function get_cond_mu_cov(t_x, t_y, y, gp_params)
    
    cov_xx = generate_covariance_matrix(t_x, gp_params)
    cov_yy = generate_covariance_matrix(t_y, gp_params)
    cov_xy = generate_cross_covariance_matrix(t_x, t_y, gp_params)
    m_x = generate_mean(t_x, gp_params[:mu])

    m_xgy = m_x + cov_xy * (cov_yy\(y - generate_mean(t_y, gp_params[:mu]))) #may be able to make more efficient by using cholesky decomposition
    cov_xgy = cov_xx - cov_xy * (cov_yy\transpose(cov_xy)) #may be able to make more efficient by using cholesky decomposition
    dt = cov_xgy - transpose(cov_xgy)
    #A = UpperTriangular(cov_xgy)
    symcov = ( cov_xgy + transpose(cov_xgy) ) ./ 2#more error: A + transpose(A) - Diagonal(cov_xgy)
    return m_xgy, symcov
    
end
