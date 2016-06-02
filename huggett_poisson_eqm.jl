#------------------------------
#   Solve for the equilibrium interest rate in the Hugget economy.
#
#   First, solve the HJB equation to get the A ("transition") matrix, then
#   solve the discretized KF equations. Then iterate over interest rates to
#   get the asset supply function. Update interest rate until convergence.
#
#   This code uses an implicit method for the solution: use
#   an 'upwind' scheme consisting of forwards and backwards difference
#   approximation to the derivative of the value function.
#
#   Assume: CRRA utility function u(c) = c^(1-s)/(1-s)
#   Note that r < rho
#
#   Written by Ben Moll et al (2015). Optimized for speed #   by SeHyoun Ahn. Translation into Julia by James Graham
#------------------------------

# Define types for workspace variables
type Huggett_model
  sigma::Float64            # risk aversion coefficient
  rho::Float64              # discount rate.
  z1::Float64               # Income in state 1
  z2::Float64               # Income in state 2
  z::Array{Float64}        # Vector for the two incomes
  la1::Float64              # State 1 switching intensity
  la2::Float64              # State 2 switching intensity
  la::Array{Float64}       # Vector for the two intensities
  I::Int64                  # Number of grids for assets
  Ir::Int64                   # Number of interest rate iterations
  amin::Float64             # Min assets = borrowing constraint
  amax::Float64
  rmin::Float64             # Min interest rate
  rmax::Float64             # Max interest rate
  agrid::Vector{Float64}    # Grid space for assets
  aa::Array{Float64}       # Matrix for asset grid space
  zz::Array{Float64}       # Matrix for income grid space
  da::Float64               # This is the implied grid step size
  maxit::Int64
  crit::Float64
  crit_S::Float64             # Tolerance for asset mkt clearing
  Delta::Float64
  Aswitch::SparseMatrixCSC{Float64,Int64}    # Transition matrix
  v0::Array{Float64}       # Initial guess for value function
  r0::Float64                 # Intial guess for interest rate
end

# Create model function
function Huggett_model(;sigma::Float64=1.5,
  rho::Float64=0.05,
  I::Int64=1000,
  Ir::Int64=40,
  amin::Float64=-0.15,
  amax::Float64=5.0,
  rmin::Float64=0.01,
  rmax::Float64=0.04,
  z1::Float64=0.1,
  z2::Float64=0.2,
  la1::Float64=1.2,
  la2::Float64=1.5,
  maxit::Int64=100,
  crit::Float64=1e-6,
  crit_S::Float64=1e-5,
  Delta::Float64=1000.0,
  r0::Float64=0.03)

  z       = hcat(z1,z2)
  la      = hcat(la1,la2)
  agrid   = collect(linspace(amin,amax,I))
  aa      = hcat(agrid, agrid)
  zz      = ones(I,1)*z
  da      = (amax-amin)/(I-1)
  v0 = hcat( (z[1] + r0.*agrid).^(1-sigma)/(1-sigma)/rho, (z[2] + r0.*agrid).^(1-sigma)/(1-sigma)/rho )

  Aswitch =  vcat( hcat(-speye(I)*la[1], speye(I)*la[1]), hcat(speye(I)*la[2],-speye(I)*la[2]) )

  Huggett_model(sigma, rho, z1, z2, z, la1, la2, la, I, Ir, amin, amax, rmin, rmax, agrid, aa, zz, da, maxit, crit, crit_S, Delta, Aswitch,  v0, r0)

end

# Creates an instance of the model
mod = Huggett_model()

# Define Utility funciton
function util(x; sigma::Float64=mod.sigma)
  u = x.^(1-sigma)/(1-sigma)
  return u
end

function utilprime(x; sigma::Float64=mod.sigma)
  uprime = x.^(-sigma)
  return uprime
end

# Define inverse of derivative of utility function
function utilprimeinv(x; sigma::Float64=mod.sigma)
  uprimeinv = x.^(-1/sigma)
  return uprimeinv
end

# Function to compute the KFE
function KFE(A::SparseMatrixCSC{Float64,Int64}; I::Int64=mod.I, da::Float64=mod.da)
  AT          = A'
  b           = zeros(2*I,1)
  i_fix       = 1    # fix one value so that matrix is not singular
  b[i_fix]    = 0.1
  row         = hcat(zeros(1,i_fix-1), 1, zeros(1,2*I-i_fix))
  AT[i_fix,:] = row
  #Solve linear system
  gg          = AT\b          # A'*gg = b --> gg = (A')^(-1)*b
  g_sum       = gg'*ones(2*I,1)*da
  gg          = gg./g_sum     # Normalize so sum(gg)*da) = 1
  g           = hcat(gg[1:I], gg[I+1:2*I])
  #check1      = g[:,1]'*ones(mod.I,1)*mod.da
  #check2      = g[:,2]'*ones(mod.I,1)*mod.da
  return g
end

# Initialization
dVf         = zeros(mod.I,2)      # Forward difference in each state
dVb         = zeros(mod.I,2)      # Backward difference in each state
c           = zeros(mod.I,2)      # Consumption in each state
A           = sparse(zeros(2*mod.I,2*mod.I))   # Transition matrix
r_r         = zeros(mod.Ir,1)     # Interest rate guesses
rmin_r      = zeros(mod.Ir,1)     # Lower bound for bisection
rmax_r      = zeros(mod.Ir,1)     # Upper bound for bisection
g_r         = zeros(mod.I,2,mod.Ir)   # Stationary distribution
adot        = zeros(mod.I,2,mod.Ir)   # Policy function
V_r         = zeros(mod.I,2,mod.Ir)   # Value function
V           = zeros(mod.I,2)        # Value function
S           = zeros(mod.Ir,1)     # Aggregate asset supply
r           = mod.r0
dist        = zeros(mod.maxit,1)
r_end       = 0                   # Last iteration value

for ir in 1:mod.Ir
  r_r[ir]     = r
  rmin_r[ir]  = mod.rmin
  rmax_r[ir]  = mod.rmax

  if ir>1
    mod.v0 = V_r[:,:,ir-1]
  end

  v = mod.v0

  for n = 1:mod.maxit
    V               = v
    # forward difference
    dVf[1:mod.I-1,:]    = (V[2:mod.I,:]-V[1:mod.I-1,:])/mod.da
    dVf[mod.I,:]        = utilprime(mod.z + r.*mod.amax)
    # backward difference
    dVb[1,:]            = utilprime(mod.z + r.*mod.amin)  #state constraint boundary condition
    dVb[2:mod.I,:]      = (V[2:mod.I,:]-V[1:mod.I-1,:])/mod.da

    # I_concave       = dVb .> dVf #indicator whether value function is concave (problems arise if this is not the case)

    #consumption and savings with forward difference
    cf              = utilprimeinv(dVf)
    ssf             = mod.zz + r.*mod.aa - cf
    #consumption and savings with backward difference
    cb              = utilprimeinv(dVb)
    ssb             = mod.zz + r.*mod.aa - cb
    #consumption and derivative of value function at steady state
    c0              = mod.zz + r.*mod.aa
    dV0             =utilprime(c0)

    # dV_upwind makes a choice of forward or backward differences based on the sign of the drift
    If              = ssf .> 0    # positive drift --> forward difference
    Ib              = ssb .< 0    # negative drift --> backward difference
    I0              = (1-If-Ib)   # at steady state
    #make sure backward difference is used at amax
    #Ib(I,:) = 1; If(I,:) = 0;
    #STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS muf > 0:
    #already taken care of automatically

    dV_Upwind       = dVf.*If + dVb.*Ib + dV0.*I0    #important to include third term
    c               = utilprimeinv(dV_Upwind)
    u               = util(c)

    #CONSTRUCT MATRIX
    X               = - min(ssb,0)/mod.da
    Y               = - max(ssf,0)/mod.da + min(ssb,0)/mod.da
    Z               = max(ssf,0)/mod.da

    A1              = spdiagm(Y[:,1],0,mod.I,mod.I) + spdiagm(X[2:mod.I,1],-1,mod.I,mod.I) + spdiagm(Z[1:mod.I-1,1],1,mod.I,mod.I)
    A2              = spdiagm(Y[:,2],0,mod.I,mod.I) + spdiagm(X[2:mod.I,2],-1,mod.I,mod.I) + spdiagm(Z[1:mod.I-1,2],1,mod.I,mod.I)
    A               = vcat(hcat(A1, sparse(zeros(mod.I,mod.I))), hcat(sparse(zeros(mod.I,mod.I)), A2)) + mod.Aswitch
    B               = (mod.rho + 1/mod.Delta)*speye(2*mod.I) - A

    u_stacked       = vcat(u[:,1], u[:,2])
    V_stacked       = vcat(V[:,1], V[:,2])

    b               = u_stacked + V_stacked/mod.Delta
    V_stacked       = B\b             # Solve linear system of eqns

    V               = hcat(V_stacked[1:mod.I], V_stacked[mod.I+1:2*mod.I])

    Vchange         = V - v
    v               = V

    dist[n] = maximum(maximum(abs(Vchange)))
    if dist[n]<mod.crit
      @printf("-----------------------\n")
      @printf("Interest rate = %f\n", r)
      @printf("Value Function Converged, Iteration = %i\n", n)
      break
    end

  end

  ## Solve the Kolmogorov Forward (Fokker-Planck) equations
  g = KFE(A)

  # Pack up output
  g_r[:,:,ir] = g
  adot[:,:,ir] = mod.zz + r.*mod.aa - c
  V_r[:,:,ir] = V
  S[ir] = (g[:,1]'*mod.agrid*mod.da + g[:,2]'*mod.agrid*mod.da)[1,1]

  #UPDATE INTEREST RATE
  r_end = ir

  if S[ir]>mod.crit_S
    @printf("Excess Supply\n")
    mod.rmax = r
    r = 0.5*(r+mod.rmin)
  elseif S[ir]<-mod.crit_S
    @printf("Excess Demand\n")
    mod.rmin = r
    r = 0.5*(r+mod.rmax)
  elseif abs(S[ir])<mod.crit_S
    @printf("Equilibrium Found, Interest rate = %f/n", r)
    break
  end

end

## EXPORT DATA FOR GRAPHING IN PYTHON
using DataFrames
df = DataFrame(adot[:,:,r_end])
writetable("adot.csv", df)

df = DataFrame(c)
writetable("cons.csv", df)

df = DataFrame(g_r[:,:,r_end])
writetable("dist.csv", df)


## GRAPHS
using Gadfly

# Savings Policy function
myplot1 = plot(layer(x=mod.agrid,y=adot[:,1,r_end],Geom.line,Theme(line_width=1.5mm)), layer(x=mod.agrid,y=adot[:,2,r_end],Geom.line, Theme(line_width=1.5mm,default_color=colorant"red")),
xintercept=[mod.amin], yintercept=[0], Geom.hline(color=colorant"black", size=0.5mm), Geom.vline(color=colorant"black", size=0.5mm),
Guide.title("Savings policy function"),
Guide.xlabel("Wealth, <i>a"),
Guide.ylabel("Savings, <i>s<sub>i</sub>(a)"),
Coord.cartesian(xmin=mod.amin, xmax=mod.amax), Theme(major_label_font_size=6mm,
minor_label_font_size=4mm))

using Cairo
using Fontconfig

# Distributions
plot(layer(x=mod.agrid,y=g_r[:,1,r_end],Geom.line,Theme(line_width=1.5mm)), layer(x=mod.agrid,y=g_r[:,2,r_end],Geom.line, Theme(line_width=1.5mm,default_color=colorant"red")),
xintercept=[mod.amin], Geom.vline(color=colorant"black", size=0.5mm), yintercept=[0], Geom.hline(color=colorant"black", size=0.5mm),
Guide.title("Stationary distributions"),
Guide.xlabel("Wealth, <i>a"),
Guide.ylabel("Density, <i>g<sub>i</sub>(a)"),
Coord.cartesian(xmin=mod.amin, xmax=1), Theme(major_label_font_size=6mm,
minor_label_font_size=4mm))
