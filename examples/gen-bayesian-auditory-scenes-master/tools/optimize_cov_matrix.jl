function exponential_kernel(ts1, ts2, sigma, scale)
    return (sigma^2)*exp(-((ts1-ts2)^2)/(2*scale^2))
end

function exponential_kernel_broadcastable(ts1, ts2, sigma, scale)
    return (sigma^2)*exp.(-((ts1 .- ts2).^2) / (2*scale^2))
end

const ts = collect(0.0:0.001:1.0)

function version1(ts, sigma, noise, scale)
    n = length(ts)
    cov = zeros(n,n)
    for i = 1:n
        for j = 1:i
            @inbounds cov[i, j] = i == j ? (sigma^2 + noise + 1e-4) : exponential_kernel(ts[i], ts[j], sigma, scale)
        end
    end
    cov
end

function version2(ts, sigma, noise, scale)
    n = length(ts)
    cov = exponential_kernel_broadcastable.(ts, ts', sigma, scale)
    for i=1:n
        cov[i,i] = sigma^2 + noise + 1e-4
    end
    cov
end

function test_version1()
    for i=1:100
        version1(ts, 0.1, 0.1, 0.1)
    end
end

function test_version2()
    for i=1:100
        version2(ts, 0.1, 0.1, 0.1)
    end
end

# precompile
@time test_version1()
@time test_version2()

# actually time it
@time test_version1()
@time test_version2()
