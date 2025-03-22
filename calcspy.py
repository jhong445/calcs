import pulp
import time


def solve_optimization_problem(max_solutions=5, optimality_tolerance=0.0):
    #allow for multiple soln
    all_solutions = []
    best_objective = float('-inf')
    
    for solution_number in range(1, max_solutions + 1):
        m = pulp.LpProblem(name="resource_allocation", sense=pulp.LpMaximize)
        
        #param
        #CBC parameters are set through solver options in pulp
        solver = pulp.PULP_CBC_CMD(msg=False)  #loglevel=0
        #idk if the othere parameters exist in pulp
        
        #constants
        total_hours = 40
        total_seconds = total_hours * 60 * 60
        menu_time = 75
        ebi_time = 74.8 + menu_time
        lnf_time = 156.3 + menu_time
        ebi_gph = 3600/ebi_time
        lnf_gph = 3600/lnf_time
        total_energy = 6000

        multi = [5, 10, 15, 19, 23, 26, 29, 31, 33, 35]
        ebi_points = [1430 * m for m in multi]
        lnf_points = [1677 * m for m in multi]
        ebi_energy = list(range(1, 11))
        lnf_energy = list(range(1, 11))
        
        #variables
        ebi_upper_bound = int(total_hours * ebi_gph)
        lnf_upper_bound = int(total_hours * lnf_gph)
        x = {}
        y = {}
        for i in range(1, 11):
            x[i] = pulp.LpVariable(f"x{i}", lowBound=0, cat='Integer')
            y[i] = pulp.LpVariable(f"y{i}", lowBound=0, cat='Integer')
        
        #func
        m += sum(ebi_points[i-1] * x[i] for i in range(1, 11)) + sum(lnf_points[i-1] * y[i] for i in range(1, 11))

        #constraints
        m += sum(x[i] for i in range(1, 11)) <= ebi_upper_bound
        m += sum(y[i] for i in range(1, 11)) <= lnf_upper_bound

        #energy
        m += (sum(ebi_energy[i-1] * x[i] for i in range(1, 11)) + 
              sum(lnf_energy[i-1] * y[i] for i in range(1, 11)) <= total_energy)

        #time
        m += (ebi_time * sum(x[i] for i in range(1, 11)) + 
              lnf_time * sum(y[i] for i in range(1, 11)) <= total_seconds)
        
        #finding additional soln within set tolerance
        if best_objective > float('-inf'):
            m += (sum(ebi_points[i-1] * x[i] for i in range(1, 11)) + 
                 sum(lnf_points[i-1] * y[i] for i in range(1, 11)) >= 
                 best_objective * (1 - optimality_tolerance))
        
        #exclusion constraints
        for (prev_x_values, prev_y_values) in all_solutions:
            #ensure the soln is different
            nonzero_x_indices = [i for i in range(1, 11) if prev_x_values[i] > 0]
            nonzero_y_indices = [i for i in range(1, 11) if prev_y_values[i] > 0]
            
            if len(nonzero_x_indices) > 0 or len(nonzero_y_indices) > 0:
                m += (sum(x[i] for i in nonzero_x_indices) + 
                     sum(y[i] for i in nonzero_y_indices) <= 
                     sum(prev_x_values[i] for i in nonzero_x_indices) + 
                     sum(prev_y_values[i] for i in nonzero_y_indices) - 20)
        
        #solve
        m.solve(solver)
        
        #results
        status = pulp.LpStatus[m.status]
        print(f"Solution {solution_number} - Status: {status}")
        
        if m.status == pulp.LpStatusOptimal:
            obj_value = pulp.value(m.objective)
            if solution_number == 1:
                best_objective = obj_value
            
            x_values = {i: pulp.value(x[i]) for i in range(1, 11)}
            y_values = {i: pulp.value(y[i]) for i in range(1, 11)}
            
            all_solutions.append((x_values, y_values))
            
            print(f"\nObjective Value: {obj_value}")
            
            print("\nebi Values (x):")
            for i in range(1, 11):
                val = x_values[i]
                if val > 0:
                    print(f"x{i} = {val} (Points: {ebi_points[i-1] * val}, Energy: {ebi_energy[i-1] * val})")
                else:
                    print(f"x{i} = {val}")
            
            print("\nlnf Values (y):")
            for i in range(1, 11):
                val = y_values[i]
                if val > 0:
                    print(f"y{i} = {val} (Points: {lnf_points[i-1] * val}, Energy: {lnf_energy[i-1] * val})")
                else:
                    print(f"y{i} = {val}")
            
            #total resources used
            energy_used = sum(ebi_energy[i-1] * x_values[i] for i in range(1, 11)) + sum(lnf_energy[i-1] * y_values[i] for i in range(1, 11))
            time_used = ebi_time * sum(x_values.values()) + lnf_time * sum(y_values.values())
            
            print("\nResource Usage Summary:")
            print(f"Energy used: {energy_used} / {total_energy} ({round(energy_used/total_energy*100, 2)}%)")
            print(f"Time used: {round(time_used/3600, 2)} hours / {total_hours} hours ({round(time_used/total_seconds*100, 2)}%)")
            
            #total score
            total_ebi_points = sum(ebi_points[i-1] * x_values[i] for i in range(1, 11))
            total_lnf_points = sum(lnf_points[i-1] * y_values[i] for i in range(1, 11))
            print("\nTotal Points:")
            print(f"ebi points: {total_ebi_points}")
            print(f"lnf points: {total_lnf_points}")
            print(f"Total points: {total_ebi_points + total_lnf_points}")
            print("\n" + "-"*50 + "\n")
        else:
            print("No more solutions found. Status: {status}")
            break
    
    print(f"Found {len(all_solutions)} solutions.")
    return all_solutions

#for debugging
import time
start_time = time.time()
solutions = solve_optimization_problem(5, 0.0)
end_time = time.time()
print(f"Execution time: {end_time - start_time} seconds")


