using JuMP, Cbc


function solve_optimization_problem(max_solutions=5, optimality_tolerance=0.0)
    #allow for multiple soln
    all_solutions = []
    best_objective = -Inf
    
    for solution_number in 1:max_solutions
        m = Model(Cbc.Optimizer)
        
        #param
        set_optimizer_attribute(m, "logLevel", 0)
        set_optimizer_attribute(m, "presolve", "on")
        set_optimizer_attribute(m, "cuts", "on")
        set_optimizer_attribute(m, "heuristics", "on")
        set_optimizer_attribute(m, "perturbation", "on")

        #constants
        total_hours = 40
        total_seconds = total_hours * 60 * 60
        menu_time = 75
        ebi_time = 74.8 + menu_time
        lnf_time = 156.3 + menu_time
        sage_time = 150.7 + menu_time
        ebi_gph = 3600/ebi_time
        lnf_gph = 3600/lnf_time
        sage_gph = 3600/sage_time
        total_energy = 6000

        multi = [5, 10, 15, 19, 23, 26, 29, 31, 33, 35]
        ebi_points = 1430 .* multi
        lnf_points = 1677 .* multi
        sage_points = 1671 .* multi
        ebi_energy = collect(1:10)
        lnf_energy = collect(1:10)
        sage_energy = collect(1:10)

        #variables
        ebi_upper_bound = floor(Int, total_hours * ebi_gph)
        lnf_upper_bound = floor(Int, total_hours * lnf_gph)
        sage_upper_bound = floor(Int, total_hours * sage_gph)
        @variable(m, 0 <= x[i=1:10], Int)
        @variable(m, 0 <= y[i=1:10], Int)
        @variable(m, 0 <= z[i=1:10], Int)
        
        #func
        @objective(m, Max, sum(ebi_points[i] * x[i] for i in 1:10) + 
                          sum(lnf_points[i] * y[i] for i in 1:10) +
                          sum(sage_points[i] * z[i] for i in 1:10))

        #constraints
        @constraint(m, sum(x) <= ebi_upper_bound)
        @constraint(m, sum(y) <= lnf_upper_bound)
        @constraint(m, sum(z) <= sage_upper_bound)

        #energy
        @constraint(m, energy_constraint, 
                    sum(ebi_energy[i] * x[i] for i in 1:10) + 
                    sum(lnf_energy[i] * y[i] for i in 1:10) +
                    sum(sage_energy[i] * z[i] for i in 1:10)<= total_energy)

        #time
        @constraint(m, time_constraint,
                    ebi_time * sum(x[i] for i in 1:10) + 
                    lnf_time * sum(y[i] for i in 1:10) +
                    sage_time * sum(z[i] for i in 1:10) <= total_seconds)
        
        #finding additional soln within set tolerance
        if best_objective > -Inf
            @constraint(m, sum(ebi_points[i] * x[i] for i in 1:10) + 
                           sum(lnf_points[i] * y[i] for i in 1:10) + 
                           sum(sage_points[i] * z[i] for i in 1:10)>= 
                           best_objective * (1 - optimality_tolerance))
        end
        
        #exclusion constraints
        for (prev_x_values, prev_y_values, prev_z_values) in all_solutions
            #ensure the soln is different
            nonzero_x_indices = findall(i -> prev_x_values[i] > 0, 1:10)
            nonzero_y_indices = findall(i -> prev_y_values[i] > 0, 1:10)
            nonzero_z_indices = findall(i -> prev_z_values[i] > 0 , 1:10)
            
            if !isempty(nonzero_x_indices) || !isempty(nonzero_y_indices) || !isempty(nonzero_z_indices)
                @constraint(m, sum(x[i] for i in nonzero_x_indices) + 
                               sum(y[i] for i in nonzero_y_indices) + 
                               sum(z[i] for i in nonzero_z_indices)<= 
                               sum(prev_x_values) + sum(prev_y_values) - 20)
            end
        end
        
        optimize!(m)
        
        #results
        status = termination_status(m)
        println("Solution $solution_number - Status: $status")
        
        if status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED
            obj_value = objective_value(m)
            if solution_number == 1
                best_objective = obj_value
            end
            
            x_values = value.(x)
            y_values = value.(y)
            z_values = value.(z)

            push!(all_solutions, (x_values, y_values, z_values))
            
            println("\nObjective Value: ", obj_value)
            
            println("\nebi Values (x):")
            for i in 1:10
                val = x_values[i]
                if val > 0
                    println("x$i = $val (Points: $(ebi_points[i] * val), Energy: $(ebi_energy[i] * val))")
                else
                    println("x$i = $val")
                end
            end
            
            println("\nlnf Values (y):")
            for i in 1:10
                val = y_values[i]
                if val > 0
                    println("y$i = $val (Points: $(lnf_points[i] * val), Energy: $(lnf_energy[i] * val))")
                else
                    println("y$i = $val")
                end
            end

            println("\nsage Values (z):")
            for i in 1:10
                val = z_values[i]
                if val > 0
                    println("z$i = $val (Points: $(sage_points[i] * val), Energy: $(sage_energy[i] * val))")
                else
                    println("z$i = $val")
                end
            end
            
            #total resources used
            energy_used = sum(ebi_energy[i] * x_values[i] for i in 1:10) + 
                         sum(lnf_energy[i] * y_values[i] for i in 1:10) +
                         sum(sage_energy[i] * z_values[i] for i in 1:10)

            time_used = ebi_time * sum(x_values) + lnf_time * sum(y_values) + sage_time * sum(z_values)
            
            println("\nResource Usage Summary:")
            println("Energy used: $energy_used / $total_energy ($(round(energy_used/total_energy*100, digits=2))%)")
            println("Time used: $(round(time_used/3600, digits=2)) hours / $total_hours hours ($(round(time_used/total_seconds*100, digits=2))%)")
            
            #total score
            total_ebi_points = sum(ebi_points[i] * x_values[i] for i in 1:10)
            total_lnf_points = sum(lnf_points[i] * y_values[i] for i in 1:10)
            total_sage_points = sum(sage_points[i] * z_values[i] for i in 1:10)
            println("\nTotal Points:")
            println("ebi points: $total_ebi_points")
            println("lnf points: $total_lnf_points")
            println("sage points: $total_sage_points")
            println("Total points: $(total_ebi_points + total_lnf_points + total_sage_points)")
            println("\n" * "-"^50 * "\n")
        else
            println("No more solutions found. Status: $status")
            break
        end
    end
    
    println("Found $(length(all_solutions)) solutions.")
    return all_solutions
end

#for debugging
@time solutions = solve_optimization_problem(5, 0.0)

