#------------------------------
#   Solve for the equilibrium interest rate in the Hugget economy with a general diffusion process.
#
#  First, solve the HJB equation to get the A ("transition") matrix, then
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
#   Written by Ben Moll et al (2015). Optimized for speed by SeHyoun Ahn. Translation into Julia by James Graham
#------------------------------


type Huggett_model
  gamma::Float64            # risk aversion coefficient
  rho::Float64              # discount rate.
  ou_var::Float64           # Ornstein Uhlenbeck variance
  zmean::Float64            # Mean of log-normal distriution for the Ornstein Uhlenbeck process
  ou_corr::Float64          # Ornstein Uhlenbeck correlation
  theta::Float64            # Ornstein Uhlenbeck param
  sig2::Float64            # Ornstein Uhlenbeck variance
  I::Int64                  # Number of grids for assets
  J::Int64                  # Number of grids for income
  Ir::Int64                 # Number of interest rate iterations
  amin::Float64             # Min assets = borrowing constraint
  amax::Float64             # Max assets
  zmin::Float64             # Min income
  zmax::Float64             # Max income
  rmin::Float64             # Min interest rate
  rmax::Float64             # Max interest rate
  agrid::Array{Float64}    # Grid space for assets
  zgrid::Array{Float64}    # Grid space for incoomes
  aa::Array{Float64}        # Matrix for asset grid space
  zz::Array{Float64}        # Matrix for income grid space
  da::Float64               # Asset grid step size
  dz::Float64               # Income grid step size
  dz2::Float64              # Income grid step size squared
  Delta::Float64            # Time step-size
  mu::Array{Float64}       # Drift from Ito's lemma
  s2::Array{Float64}       # Std Dev from Ito's lemma
  chi::Array{Float64}      # elements in the transition matrix
  nu::Array{Float64}       # elements in the transition matrix
  zeta::Array{Float64}     # elements in the transition matrix
  Cswitch::SparseMatrixCSC{Float64,Int64}  # Transition matrix
  v0::Array{Float64}       # Initial guess for value function
  r0::Float64                 # Intial guess for interest rate
  maxit::Int64              # Max iterations for solving HJB
  crit::Float64
  crit_S::Float64           # Tolerance for asset mkt clearing
end

# Create model function
function Huggett_model(;gamma::Float64=2.0,
  rho::Float64=0.05,
  ou_var::Float64=0.1,
  ou_corr::Float64=0.5,
  I::Int64=100,
  J::Int64=100,
  Ir::Int64=40,
  amin::Float64=-0.15,
  amax::Float64=2.0,
  rmin::Float64=-0.05,
  rmax::Float64=0.045,
  maxit::Int64=100,
  crit::Float64=1e-6,
  crit_S::Float64=1e-5,
  Delta::Float64=1000.0,
  r0::Float64=0.03)

  ## Ornstein-Uhlenbeck: dlog(z) = -theta*log(z)dt + sig*dW
  zmean   = exp(ou_var)
  theta   = -log(ou_corr)
  sig2    = theta*ou_var
  ## Ornstein-Uhlenbeck: dz = theta(zmean - z)dt + sig*dW
  #theta   = -log(ou_corr)
  #ou_var  = sig2/(theta)
  #zmean   = 1

  zmin    = zmean*0.65
  zmax    = zmean*1.25
  agrid   = reshape(collect(linspace(amin,amax,I)),I,1)
  zgrid   = reshape(collect(linspace(zmin,zmax,J)),J,1)
  aa      = repmat(agrid,1,J)
  zz      = repmat(zgrid',I,1)
  da      = (amax-amin)/(I-1)
  dz      = (zmax-zmin)/(J-1)
  dz2     = dz^2
  ## Ornstein-Uhlenbeck: in logs
  mu      = (-theta*log(zgrid) + sig2/2).*zgrid
  s2      = sig2.*zgrid.^2
  ## Ornstein-Uhlenbeck: in LEVELS
  #mu      = theta*(zmean - zgrid)
  #s2      = sig2.*ones(1,J)

  chi     = s2/(2*dz2)
  nu      = - s2/dz2 - mu/dz
  zeta    = mu/dz + s2/(2*dz2)

  v0      = (zz + r0.*aa).^(1-gamma)/(1-gamma)/rho

  #This will be the upperdiagonal of the C_switch
  updiag    = zeros(I,1)
  for j in 1:J-1
      updiag=vcat(updiag, repmat([zeta[j]],I,1))
  end

  #This will be the center diagonal of the B_switch
  centdiag = repmat([chi[1]+nu[1]],I,1)
  for j in 2:J-1
      centdiag = vcat(centdiag, repmat([nu[j]],I,1))
  end
  centdiag = vcat(centdiag, repmat([nu[J]+zeta[J]],I,1))

  #This will be the lower diagonal of the B_switch
  lowdiag = repmat([chi[2]],I,1)
  for j in 3:J
      lowdiag = vcat(lowdiag, repmat([chi[j]],I,1))
  end

  # Add upper, center, and lower diagonal into a sparse matrix
  Cswitch     = spdiagm(centdiag[:,1],0,I*J,I*J) + spdiagm(lowdiag[:,1],-I,I*J,I*J) + spdiagm(updiag[I+1:end,1],I,I*J,I*J)

  Huggett_model(gamma, rho, ou_var, zmean, ou_corr, theta, sig2, I, J, Ir, amin, amax, zmin, zmax, rmin, rmax, agrid, zgrid, aa, zz, da, dz, dz2, Delta, mu, s2, chi, nu, zeta, Cswitch, v0, r0, maxit, crit, crit_S)
end

# Creates an instance of the model
mod = Huggett_model()

# Define Utility funciton
function util(x; gamma::Float64=mod.gamma)
  u = x.^(1-gamma)/(1-gamma)
  return u
end

# Define derivative of Utility funciton
function utilprime(x; gamma::Float64=mod.gamma)
  uprime = x.^(-gamma)
  return uprime
end

# Define inverse of derivative of utility function
function utilprimeinv(x; gamma::Float64=mod.gamma)
  uprimeinv = x.^(-1/gamma)
  return uprimeinv
end

# Function to compute the KFE
function KFE(A::SparseMatrixCSC{Float64,Int64}; I::Int64=mod.I, J::Int64=mod.J, da::Float64=mod.da, dz::Float64=mod.dz)
  AT          = A'
  b           = zeros(I*J,1)
  i_fix       = 1    # fix one value so that matrix is not singular
  b[i_fix]    = 0.1
  row         = hcat(zeros(1,i_fix-1), 1, zeros(1,I*J-i_fix))
  AT[i_fix,:] = row
  #Solve linear system
  gg          = AT\b          # A'*gg = b --> gg = (A')^(-1)*b
  g_sum       = gg'*ones(I*J,1)*da*dz
  gg          = gg./g_sum     # Normalize so sum(gg)*da) = 1
  g           = reshape(gg,I,J)
  return g
end

# Initialization of value functions and consumption policy
Vaf         = zeros(mod.I,mod.J)
Vab         = zeros(mod.I,mod.J)
Vzf         = zeros(mod.I,mod.J)
Vzb         = zeros(mod.I,mod.J)
Vzz         = zeros(mod.I,mod.J)
c           = zeros(mod.I,mod.J)
r_r         = zeros(mod.Ir,1)
rmin_r      = zeros(mod.Ir,1)
rmax_r      = zeros(mod.Ir,1)
g_r         = zeros(mod.I,mod.J,mod.Ir)     # Stationary distribution
adot        = zeros(mod.I,mod.J,mod.Ir)    # Policy function
V_r         = zeros(mod.I,mod.J,mod.Ir)    # Value function
S           = zeros(mod.Ir,1)      # Aggregate asset supply
V           = zeros(mod.I,2)        # Value function
A           = sparse(zeros(mod.J*mod.I,mod.J*mod.I))   # Transition matrix
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

  for n in 1:mod.maxit
    V               = v
    # forward difference
    Vaf[1:mod.I-1,:]    = (V[2:mod.I,:]-V[1:mod.I-1,:])/mod.da
    Vaf[mod.I,:]        = utilprime(mod.zgrid + r.*mod.amax)
    # backward difference
    Vab[1,:]            = utilprime(mod.zgrid + r.*mod.amin)  #state constraint boundary condition
    Vab[2:mod.I,:]      = (V[2:mod.I,:]-V[1:mod.I-1,:])/mod.da

    #consumption and savings with forward difference
    cf              = utilprimeinv(Vaf)
    sf             = mod.zz + r.*mod.aa - cf
    #consumption and savings with backward difference
    cb              = utilprimeinv(Vab)
    sb             = mod.zz + r.*mod.aa - cb
    #consumption and derivative of value function at steady state
    c0              = mod.zz + r.*mod.aa
    Va0             =utilprime(c0)

    # dV_upwind makes a choice of forward or backward differences based on the sign of the drift
    If              = sf .> 0    # positive drift --> forward difference
    Ib              = sb .< 0    # negative drift --> backward difference
    I0              = (1-If-Ib)   # at steady state
    #make sure backward difference is used at amax
    #Ib(I,:) = 1; If(I,:) = 0;
    #STATE CONSTRAINT at amin: USE BOUNDARY CONDITION UNLESS muf > 0:
    #already taken care of automatically

    Va_Upwind       = Vaf.*If + Vab.*Ib + Va0.*I0    #important to include third term
    c               = utilprimeinv(Va_Upwind)
    u               = util(c)

    #CONSTRUCT MATRIX
    X               = - min(sb,0)/mod.da
    Y               = - max(sf,0)/mod.da + min(sb,0)/mod.da
    Z               = max(sf,0)/mod.da

    updiag          = 0
    for j = 1:mod.J
        updiag      = vcat(updiag, Z[1:mod.I-1,j], 0)      # Note: Z_I,j = 0 for all j
    end
    updiag = updiag[2:end-1,1]

    centdiag = reshape(Y,mod.I*mod.J,1)

    lowdiag=X[2:mod.I,1]
    for j=2:mod.J
        lowdiag=vcat(lowdiag, 0, X[2:mod.I,j])          # Note: X_1,j = 0 for all j
    end

    B              = spdiagm(centdiag[:,1],0,mod.I*mod.J,mod.I*mod.J) + spdiagm(updiag[:,1],1,mod.I*mod.J,mod.I*mod.J) + spdiagm(lowdiag[:,1],-1,mod.I*mod.J,mod.I*mod.J)

    A              = B + mod.Cswitch
    InvMat         = (1/mod.Delta + mod.rho)*speye(mod.I*mod.J) - A

    u_stacked      = reshape(u,mod.I*mod.J,1)
    V_stacked      = reshape(V,mod.I*mod.J,1)

    b              = u_stacked + V_stacked/mod.Delta
    V_stacked      = InvMat\b  #SOLVE SYSTEM OF EQUATIONS
    V              = reshape(V_stacked,mod.I,mod.J)
    Vchange        = V - v
    v              = V

    dist[n] = maximum(maximum(abs(Vchange)))
    if dist[n]<mod.crit
      @printf("-----------------------\n")
      @printf("Interest rate = %f\n", r)
      @printf("Value Function Converged, Iteration = %i\n", n)
      break
    end
  end


  # Solve Kolmogorov Forward equation (Fokker-Plankck equation)
  g = KFE(A)
  g_r[:,:,ir]     = g
  adot[:,:,ir]    = mod.zz + r.*mod.aa - c
  V_r[:,:,ir]     = V

  S[ir]           = ((g'*mod.agrid*mod.da)'*ones(mod.I,1)*mod.dz)[1,1]

  #UPDATE INTEREST RATE
  r_end = ir

  if S[ir]>mod.crit_S
    @printf("Excess Supply at r = %0.2f\n", r)
    mod.rmax = r
    r = 0.5*(r+mod.rmin)
  elseif S[ir]<-mod.crit_S
    @printf("Excess Demand at r = %0.2f\n", r)
    mod.rmin = r
    r = 0.5*(r+mod.rmax)
  elseif abs(S[ir])<mod.crit_S
    @printf("Equilibrium Found, Interest rate = %0.2f\n", r)
    break
  end

end


## EXPORT DATA FOR GRAPHING IN PYTHON
using DataFrames
df = DataFrame(adot[:,:,r_end])
writetable("adot_diff.csv", df)

df = DataFrame(c)
writetable("cons_diff.csv", df)

df = DataFrame(g_r[:,:,r_end])
writetable("dist_diff.csv", df)


## GRAPHS
using Gadfly

# Distribution using a contour plot
plot(z=adot[:,:,r_end], x = mod.agrid, y = mod.zgrid, Geom.contour, Theme(line_width=1mm),Guide.title("Stationary distribution"),
Guide.xlabel("Wealth, <i>a"),
Guide.ylabel("Income, <i>z"),
Coord.cartesian(xmin=mod.amin, xmax=2), Theme(major_label_font_size=6mm,
minor_label_font_size=4mm))

# Savings Policy function
plot(layer(x=mod.agrid,y=adot[:,1,r_end],Geom.line,Theme(line_width=1.5mm)), layer(x=mod.agrid,y=adot[:,2,r_end],Geom.line, Theme(line_width=1.5mm,default_color=colorant"red")),
xintercept=[mod.amin], yintercept=[0], Geom.hline(color=colorant"black", size=0.5mm), Geom.vline(color=colorant"black", size=0.5mm),
Guide.title("Savings policy function"),
Guide.xlabel("Wealth, <i>a"),
Guide.ylabel("Savings, <i>s<sub>i</sub>(a)"),
Coord.cartesian(xmin=mod.amin, xmax=mod.amax), Theme(major_label_font_size=6mm,
minor_label_font_size=4mm))


# Distributions
plot(layer(x=mod.agrid,y=g_r[:,50,r_end],Geom.line,Theme(line_width=1.5mm)), layer(x=mod.agrid,y=g_r[:,60,r_end],Geom.line, Theme(line_width=1.5mm,default_color=colorant"red")),
xintercept=[mod.amin], Geom.vline(color=colorant"black", size=0.5mm), yintercept=[0], Geom.hline(color=colorant"black", size=0.5mm),
Guide.title("Stationary distributions"),
Guide.xlabel("Wealth, <i>a"),
Guide.ylabel("Density, <i>g<sub>i</sub>(a)"),
Coord.cartesian(xmin=mod.amin, xmax=1), Theme(major_label_font_size=6mm,
minor_label_font_size=4mm))
