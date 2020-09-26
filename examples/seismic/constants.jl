struct Range
    min::Float64
    max::Float64
end

const NUM_STATIONS = 5
const EVENT_MAG_MIN = 2
const EVENT_MAG_EXP_RATE = log(10)
const SPACE_RANGE = Range(0, 1)
const TIME_RANGE = Range(0, 10)
const α_I = 20; const β_I = 2
const α_N = 3;  const β_N = 2
const α_F = 20; const β_F = 1
const μ_V = 5;  const σ2_V = 1
const μ_B = 2;  const σ2_B = 1
const μ_ν = 1;  const σ2_ν = 1
const α_S = 2; const β_S = 1
const μ_t = 0; const λ_t = 1000; const α_t = 20; const β_t = 1
const μ_a = 0; const λ_a = 1; const α_a = 2; const β_a = 1
const μ_n = 0; const λ_n = 1; const α_n = 2; const β_n = 1
