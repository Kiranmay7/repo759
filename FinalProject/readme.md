The Lattice Boltzmann Method (LBM) is a numerical approach for simulating fluid dynamics, particularly useful for Computational Fluid Dynamics (CFD) problems.
Boltzmann transport equation
The most common collision model is the BGK (Bhatnagar-Gross-Krook) approximation, simplifying the equation to:

𝑓𝑖(𝑥+𝑒𝑖Δ𝑡,𝑡+Δ𝑡)−𝑓𝑖(𝑥,𝑡)=−1𝜏(𝑓𝑖(𝑥,𝑡)−𝑓𝑖eq(𝑥,𝑡))fi​(x+eiΔt,t+Δt)−fi(x,t)=−τ1(fi(x,t)−fieq(x,t))
fi represents the distribution function in a given lattice direction 𝑖.
ei is the discrete velocity vector corresponding to direction 𝑖.
fieq is the equilibrium distribution function
