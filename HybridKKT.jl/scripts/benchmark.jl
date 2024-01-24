
using MadNLPHSL

function solve_hsl(nlp; options...)
    solver_sparse = MadNLPSolver(
        nlp;
        linear_solver=Ma27Solver,
        options...
    )
    MadNLP.solve!(solver_sparse)
    return solver_sparse
end

function solve_sparse_condensed(nlp; options...)
    solver = MadNLPSolver(
        nlp;
        kkt_system=MadNLP.SparseCondensedKKTSystem,
        equality_treatment=MadNLP.RelaxEquality,
        fixed_variable_treatment=MadNLP.RelaxBound,
        options...
    )
    MadNLP.solve!(solver)
    return solver
end

function solve_hybrid(nlp; gamma=1e2, options...)
    solver = MadNLPSolver(
        nlp;
        kkt_system=HybridCondensedKKTSystem,
        equality_treatment=MadNLP.EnforceEquality,
        fixed_variable_treatment=MadNLP.MakeParameter,
        options...
    )
    solver.kkt.gamma[] = gamma
    MadNLP.solve!(solver)
    return solver
end
