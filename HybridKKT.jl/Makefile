
JULIAEXEC=julia

install: 
	$(JULIAEXEC) --project -e "using Pkg; Pkg.instantiate()" 

clean: 
	rm -rf results/

build:
	zip -r results.zip results/

update:
	$(JULIAEXEC) --project -e "using Pkg; Pkg.update()"  
	
tests:
	$(JULIAEXEC) --project benchmarks/COPS/benchmark.jl --quick  
	$(JULIAEXEC) --project benchmarks/OPF/benchmark.jl --quick 

table1: 
	$(JULIAEXEC) --project benchmarks/OPF/benchmark_linear_solvers.jl 

table2:
	$(JULIAEXEC) --project benchmarks/OPF/relaxation_convergence.jl 

figure1:
	$(JULIAEXEC) --project benchmarks/OPF/hybrid_convergence.jl 

figure2:
	$(JULIAEXEC) --project benchmarks/OPF/benchmark_kkt.jl 

opf_benchmark:
	$(JULIAEXEC) --project benchmarks/OPF/benchmark.jl 

cops_benchmark:
	$(JULIAEXEC) --project benchmarks/COPS/benchmark.jl 

benchmarks: opf_benchmark cops_benchmark

article: table1 figure1 figure2 

all: benchmarks article

