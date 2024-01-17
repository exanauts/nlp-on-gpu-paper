
using PowerModels
using ExaModels
using JuMP

PowerModels.silence()

include("model.jl")

function solve_hsl(nlp)
    solver_sparse = MadNLPSolver(
        nlp;
        linear_solver=Ma27Solver,
        print_level=MadNLP.INFO,
        max_iter=1000,
        nlp_scaling=true,
        tol=1e-4,
    )
    MadNLP.solve!(solver_sparse)
    return solver_sparse
end

function solve_sparse_condensed(nlp)
    solver = MadNLPSolver(
        nlp;
        linear_solver=LapackCPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        print_level=MadNLP.INFO,
        max_iter=1000,
        nlp_scaling=true,
        tol=1e-4,
    )
    MadNLP.solve!(solver)
    return solver
end

function solve_hybrid(nlp)
    solver = MadNLPSolver(
        nlp;
        linear_solver=LapackCPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        kkt_system=HybridCondensedKKTSystem,
        print_level=MadNLP.DEBUG,
        max_iter=1,
        nlp_scaling=true,
        tol=1e-4,
    )
    MadNLP.solve!(solver)
    return solver
end

case = "/home/fpacaud/dev/pglib-opf/pglib_opf_case118_ieee.m"
# case = "/home/fpacaud/dev/matpower/data/case9.m"
nlp = ac_power_model(case)

# solver = solve_hybrid(nlp)


    solver = MadNLPSolver(
        nlp;
        linear_solver=LapackCPUSolver,
        lapack_algorithm=MadNLP.CHOLESKY,
        kkt_system=HybridCondensedKKTSystem,
        print_level=MadNLP.DEBUG,
        max_iter=1,
        nlp_scaling=true,
        tol=1e-4,
    )
    MadNLP.solve!(solver)

    x = solver.p
    # x.values .= 1
    MadNLP.solve_refine_wrapper!(solver.d, solver, x, solver._w4)
    mul!(solver._w4, solver.kkt, solver.d)
