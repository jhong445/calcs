using LinearAlgebra, Statistics, SparseArrays

# State representation for cards: 0 to 7 (binary representation of which 3 cards we have)
# 0: [0,0,0], 1: [1,0,0], 2: [0,1,0], 3: [1,1,0], 4: [0,0,1], 5: [1,0,1], 6: [0,1,1], 7: [1,1,1]

function cards_to_state(cards)
    return cards[1] + 2*cards[2] + 4*cards[3]
end

function state_to_cards(state)
    return [state & 1, (state >> 1) & 1, (state >> 2) & 1]
end

function get_transition_probabilities(pull_in_phase, p_four_star, p_rate_up)
    """
    Get transition probabilities for regular pulls within a phase
    pull_in_phase: 0-98 (position within the phase, excluding the final special pull)
    """
    # Base probabilities for regular pulls
    p_card1 = p_rate_up
    p_card2 = p_rate_up  
    p_card3 = p_rate_up
    p_nothing = 1 - 3*p_rate_up
    
    return p_card1, p_card2, p_card3, p_nothing
end

function apply_phase_end_mechanics(cards_distribution, phase_num, p_four_star, p_rate_up)
    """
    Apply the special mechanics at the end of each phase
    Returns: (completion_prob, new_distribution_for_next_phase)
    """
    p_100 = p_rate_up / p_four_star
    is_200_phase = (phase_num % 2 == 0)  # Phases 2, 4, 6
    is_spark_phase = (phase_num % 3 == 0)  # Phases 3, 6
    sparks_before_phase_end = (phase_num - 1) ÷ 3  # Sparks available before this phase's mechanics
    
    completion_prob = 0.0
    new_distribution = zeros(8)
    
    for cards_state in 0:7
        if cards_distribution[cards_state + 1] > 0
            cards = state_to_cards(cards_state)
            entry_prob = cards_distribution[cards_state + 1]
            
            # Check if already completed with existing sparks
            if sum(cards) + sparks_before_phase_end >= 3
                completion_prob += entry_prob
                continue
            end
            
            if is_200_phase
                # 200-pull pity: guaranteed rate-up card
                # Each of the 3 cards has 1/3 probability
                for new_card_idx in 1:3
                    new_cards = copy(cards)
                    new_cards[new_card_idx] = 1
                    
                    # Check completion after getting this card + existing sparks
                    total_cards = sum(new_cards) + sparks_before_phase_end
                    
                    # Add spark if this is also a spark phase
                    if is_spark_phase
                        total_cards += 1
                    end
                    
                    if total_cards >= 3
                        completion_prob += entry_prob * (1/3)
                    else
                        new_state = cards_to_state(new_cards)
                        new_distribution[new_state + 1] += entry_prob * (1/3)
                    end
                end
            else
                # 100-pull pity: higher rate-up probability, but not guaranteed
                p_card1 = p_100
                p_card2 = p_100
                p_card3 = p_100
                p_nothing = 1 - 3*p_100
                
                # Possible outcomes
                outcomes = [
                    (cards, p_nothing),  # Nothing new
                    ([max(cards[1], 1), cards[2], cards[3]], p_card1),  # Get card 1
                    ([cards[1], max(cards[2], 1), cards[3]], p_card2),  # Get card 2
                    ([cards[1], cards[2], max(cards[3], 1)], p_card3)   # Get card 3
                ]
                
                for (new_cards, prob) in outcomes
                    if prob > 0
                        # Check completion after this outcome + existing sparks
                        total_cards = sum(new_cards) + sparks_before_phase_end
                        
                        # Add spark if this is also a spark phase
                        if is_spark_phase
                            total_cards += 1
                        end
                        
                        if total_cards >= 3
                            completion_prob += entry_prob * prob
                        else
                            new_state = cards_to_state(new_cards)
                            new_distribution[new_state + 1] += entry_prob * prob
                        end
                    end
                end
            end
        end
    end
    
    return completion_prob, new_distribution
end

function build_phase_transition_matrix(phase_num, p_four_star, p_rate_up)
    """
    Build transition matrix for regular pulls within a phase (pulls 1-99 of the phase)
    The 100th pull is handled separately by apply_phase_end_mechanics
    """
    # Each phase: 8 card states × 99 pull positions = 792 states + exit states
    # Exit states represent reaching the end of regular pulls (before phase-end mechanics)
    n_regular_states = 8 * 99
    n_exit_states = 8  # One for each possible card state at phase end
    n_states = n_regular_states + n_exit_states + 1  # +1 for absorbing state
    
    absorbing_state = n_states
    exit_states_start = n_regular_states + 1
    
    # Calculate available sparks at start of this phase
    sparks_available = (phase_num - 1) ÷ 3
    
    I = Int[]
    J = Int[]
    V = Float64[]
    
    for cards_state in 0:7
        cards = state_to_cards(cards_state)
        
        # Check if already completed with available sparks
        if sum(cards) + sparks_available >= 3
            # All states for this card configuration go directly to absorbing state
            for pull_in_phase in 0:98
                from_state = cards_state * 99 + pull_in_phase + 1
                push!(I, from_state)
                push!(J, absorbing_state)
                push!(V, 1.0)
            end
            continue
        end
        
        for pull_in_phase in 0:98
            from_state = cards_state * 99 + pull_in_phase + 1
            
            p_card1, p_card2, p_card3, p_nothing = get_transition_probabilities(
                pull_in_phase, p_four_star, p_rate_up)
            
            # Check if this is the last regular pull (pull 99 of phase)
            is_last_regular_pull = (pull_in_phase == 98)
            
            # Possible outcomes
            new_cards_sets = [
                cards,  # Nothing new
                [max(cards[1], 1), cards[2], cards[3]],  # Get card 1
                [cards[1], max(cards[2], 1), cards[3]],  # Get card 2
                [cards[1], cards[2], max(cards[3], 1)]   # Get card 3
            ]
            
            probs = [p_nothing, p_card1, p_card2, p_card3]
            
            for (i, new_cards) in enumerate(new_cards_sets)
                if probs[i] > 0
                    # Check if completed with current cards + available sparks
                    if sum(new_cards) + sparks_available >= 3
                        push!(I, from_state)
                        push!(J, absorbing_state)
                        push!(V, probs[i])
                    else
                        if is_last_regular_pull
                            # Transition to exit state for phase-end processing
                            exit_state = exit_states_start + cards_to_state(new_cards)
                            push!(I, from_state)
                            push!(J, exit_state)
                            push!(V, probs[i])
                        else
                            # Continue within phase
                            new_cards_state = cards_to_state(new_cards)
                            to_state = new_cards_state * 99 + (pull_in_phase + 1) + 1
                            push!(I, from_state)
                            push!(J, to_state)
                            push!(V, probs[i])
                        end
                    end
                end
            end
        end
    end
    
    # Exit states are terminal within this phase matrix
    for i in 1:n_exit_states
        exit_state = exit_states_start + i - 1
        push!(I, exit_state)
        push!(J, exit_state)
        push!(V, 1.0)
    end
    
    # Absorbing state transitions to itself
    push!(I, absorbing_state)
    push!(J, absorbing_state)
    push!(V, 1.0)
    
    return sparse(I, J, V, n_states, n_states), exit_states_start, absorbing_state
end

function analyze_single_phase(phase_num, entry_distribution, p_four_star, p_rate_up)
    """
    Analyze a single phase with proper phase-end mechanics
    Returns: (completion_prob_within_phase, completion_prob_at_phase_end, exit_distribution, expected_pulls_within_phase)
    """
    P, exit_states_start, absorbing_state = build_phase_transition_matrix(phase_num, p_four_star, p_rate_up)
    
    n_regular_states = 8 * 99
    transient_states = 1:n_regular_states
    
    Q = P[transient_states, transient_states]
    R_absorb = P[transient_states, absorbing_state:absorbing_state]
    R_exit = P[transient_states, exit_states_start:(exit_states_start + 7)]
    
    # Fundamental matrix for regular pulls
    N = inv(Matrix(I - Q))
    
    # Expected steps and absorption probabilities
    expected_steps = N * ones(length(transient_states))
    prob_absorb_within = (N * R_absorb)[:]
    prob_exit_to_phase_end = N * R_exit
    
    # Calculate metrics weighted by entry distribution
    completion_within_phase = 0.0
    weighted_expected_pulls = 0.0
    phase_end_distribution = zeros(8)
    
    sparks_available = (phase_num - 1) ÷ 3
    
    for cards_state in 0:7
        entry_prob = entry_distribution[cards_state + 1]
        if entry_prob > 0
            cards = state_to_cards(cards_state)
            
            # Check if immediately completed due to sparks
            if sum(cards) + sparks_available >= 3
                completion_within_phase += entry_prob
                continue
            end
            
            initial_state = cards_state * 99 + 1
            if initial_state <= length(transient_states)
                # Probability of completing within regular pulls
                prob_complete_within = prob_absorb_within[initial_state]
                completion_within_phase += entry_prob * prob_complete_within
                
                if prob_complete_within > 0
                    weighted_expected_pulls += entry_prob * prob_complete_within * expected_steps[initial_state]
                end
                
                # Distribution reaching phase end
                for exit_cards_state in 0:7
                    prob_reach_exit = prob_exit_to_phase_end[initial_state, exit_cards_state + 1]
                    phase_end_distribution[exit_cards_state + 1] += entry_prob * prob_reach_exit
                end
            end
        end
    end
    
    # Normalize expected pulls
    if completion_within_phase > 0
        weighted_expected_pulls /= completion_within_phase
    end
    
    # Apply phase-end mechanics
    completion_at_phase_end, next_phase_distribution = apply_phase_end_mechanics(
        phase_end_distribution, phase_num, p_four_star, p_rate_up)
    
    return completion_within_phase, completion_at_phase_end, next_phase_distribution, weighted_expected_pulls
end

function analyze_six_phase_system(p_four_star, p_rate_up)
    """
    Analyze the complete 6-phase system with proper phase transitions
    """
    println("Six-Phase Gacha Analysis with Proper Phase Transitions")
    println("="^62)
    
    # Initial distribution: no cards
    current_distribution = zeros(8)
    current_distribution[1] = 1.0  # State 0: [0,0,0]
    
    phase_results = []
    cumulative_completion_prob = 0.0
    total_expected_pulls = 0.0
    
    for phase in 1:6
        println("\nPhase $(phase): Pulls $((phase-1)*100 + 1)-$(phase*100)")
        available_sparks = (phase - 1) ÷ 3
        is_200_phase = (phase % 2 == 0)
        is_spark_phase = (phase % 3 == 0)
        
        println("  Available sparks at start: $(available_sparks)")
        println("  Phase type: $(is_200_phase ? "200-pull pity" : "100-pull pity")$(is_spark_phase ? " + spark reward" : "")")
        
        # Show entry distribution
        if sum(current_distribution) > 0.001
            println("  Entry distribution:")
            card_names = ["[0,0,0]", "[1,0,0]", "[0,1,0]", "[1,1,0]", "[0,0,1]", "[1,0,1]", "[0,1,1]", "[1,1,1]"]
            for (i, prob) in enumerate(current_distribution)
                if prob > 0.01
                    println("    $(card_names[i]): $(round(prob * 100, digits=1))%")
                end
            end
        end
        
        # Analyze this phase
        completion_within, completion_at_end, exit_dist, expected_within = analyze_single_phase(
            phase, current_distribution, p_four_star, p_rate_up)
        
        total_phase_completion = completion_within + completion_at_end
        
        # Calculate contribution to total expected pulls
        remaining_prob = 1 - cumulative_completion_prob
        if remaining_prob > 0
            # Pulls for those completing within phase (regular pulls 1-99)
            if completion_within > 0
                within_contribution = remaining_prob * completion_within * (expected_within + (phase-1)*100)
                total_expected_pulls += within_contribution
            end
            
            # Pulls for those completing at phase end (exactly 100 pulls in phase)
            if completion_at_end > 0
                end_contribution = remaining_prob * completion_at_end * (phase * 100)
                total_expected_pulls += end_contribution
            end
        end
        
        # Update cumulative completion
        cumulative_completion_prob += remaining_prob * total_phase_completion
        
        # Store results
        phase_info = (
            phase = phase,
            completion_within = completion_within,
            completion_at_end = completion_at_end,
            total_completion = total_phase_completion,
            cumulative_completion = cumulative_completion_prob,
            expected_within = expected_within,
            exit_distribution = exit_dist
        )
        push!(phase_results, phase_info)
        
        # Print phase results
        println("  Completion within phase (pulls 1-99): $(round(completion_within * 100, digits=2))%")
        println("  Completion at phase end (pull 100): $(round(completion_at_end * 100, digits=2))%")
        println("  Total phase completion: $(round(total_phase_completion * 100, digits=2))%")
        println("  Cumulative completion: $(round(cumulative_completion_prob * 100, digits=2))%")
        if completion_within > 0
            println("  Expected pulls if completing within: $(round(expected_within, digits=2))")
        end
        
        # Update distribution for next phase
        current_distribution = exit_dist
        
        # Break if essentially everyone has completed
        if cumulative_completion_prob > 0.9999
            println("  >>> Nearly 100% completion reached <<<")
            break
        end
    end
    
    # Handle any remaining probability (should be essentially zero)
    if cumulative_completion_prob < 1.0
        remaining = 1 - cumulative_completion_prob
        total_expected_pulls += remaining * 600  # Worst case
        println("\nRemaining probability after all phases: $(round(remaining * 100, digits=6))%")
    end
    
    return phase_results, total_expected_pulls, cumulative_completion_prob
end

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

# Parameters
p_four_star = 0.03
p_rate_up = 0.004

println("Six-Phase Gacha Markov Chain Analysis")
println("="^37)
println("Parameters:")
println("  4-star probability: $(p_four_star)")
println("  Rate-up probability: $(p_rate_up)")
println("  Phase structure:")
println("    Phase 1 (1-100): 100-pull pity")
println("    Phase 2 (101-200): 200-pull pity") 
println("    Phase 3 (201-300): 100-pull pity + spark")
println("    Phase 4 (301-400): 200-pull pity (1 spark available)")
println("    Phase 5 (401-500): 100-pull pity (1 spark available)")
println("    Phase 6 (501-600): 200-pull pity + spark (1→2 sparks)")
println()

phase_results, total_expected, final_completion = analyze_six_phase_system(p_four_star, p_rate_up)

println("\n" * "="^80)
println("SUMMARY RESULTS")
println("="^80)
println("Overall expected pulls: $(round(total_expected, digits=2))")
println("Final completion probability: $(round(final_completion * 100, digits=6))%")

println("\nDetailed Phase Breakdown:")
println("Phase | Range     | Within% | AtEnd% | Total% | Cumul% | AvgWithin")
println("------|-----------|---------|--------|--------|--------|----------")
for result in phase_results
    phase = result.phase
    range_str = "$((phase-1)*100 + 1)-$(phase*100)"
    within_pct = round(result.completion_within * 100, digits=2)
    end_pct = round(result.completion_at_end * 100, digits=2)
    total_pct = round(result.total_completion * 100, digits=2)
    cumul_pct = round(result.cumulative_completion * 100, digits=2)
    avg_within = result.completion_within > 0 ? round(result.expected_within, digits=1) : "-"
    
    println("  $(phase)   | $(lpad(range_str, 9)) | $(lpad(within_pct, 7))% | $(lpad(end_pct, 6))% | $(lpad(total_pct, 6))% | $(lpad(cumul_pct, 6))% | $(lpad(avg_within, 8))")
end

# Verification with simulation
println("\n" * "="^80)
println("VERIFICATION WITH SIMULATION")
println("="^80)

function simulate_gacha_proper(p_four_star, p_rate_up)
    p_100 = p_rate_up / p_four_star
    rate_up_cards = [0, 0, 0]
    num_pulls = 0
    spark = 0
    
    while sum(rate_up_cards) != 3 && num_pulls < 600
        num_pulls += 1
        x = rand(Float64)
        
        # Regular pull mechanics
        if x <= p_rate_up
            rate_up_cards[1] = 1
        elseif p_rate_up < x <= 2*p_rate_up
            rate_up_cards[2] = 1
        elseif 2*p_rate_up < x <= 3*p_rate_up
            rate_up_cards[3] = 1
        end
        
        # 100-pull pity (at pulls 100, 300, 500)
        if mod(num_pulls, 100) == 0 && mod(num_pulls, 200) != 0
            y = rand(Float64)
            if y <= p_100
                rate_up_cards[1] = 1
            elseif p_100 < y <= 2*p_100
                rate_up_cards[2] = 1
            elseif 2*p_100 < y <= 3*p_100
                rate_up_cards[3] = 1
            end
        end
        
        # 200-pull pity (at pulls 200, 400, 600)
        if mod(num_pulls, 200) == 0
            z = rand(Float64)
            if z <= 1/3
                rate_up_cards[1] = 1
            elseif 1/3 < z <= 2/3
                rate_up_cards[2] = 1
            else
                rate_up_cards[3] = 1
            end
        end
        
        # Spark every 300 pulls (at pulls 300, 600)
        if mod(num_pulls, 300) == 0
            spark += 1
        end
        
        # Check completion
        if sum(rate_up_cards) + spark >= 3
            rate_up_cards = [1, 1, 1]
            break
        end
    end
    
    return num_pulls
end

n_sims = 100000
println("Running $(n_sims) simulations...")
sim_results = [simulate_gacha_proper(p_four_star, p_rate_up) for _ in 1:n_sims]
sim_mean = mean(sim_results)

# Count completions by phase
phase_completions = zeros(Int, 6)
for result in sim_results
    for phase in 1:6
        if result <= phase * 100
            phase_completions[phase] += 1
            break
        end
    end
end

# Convert to percentages
phase_percentages = phase_completions ./ n_sims * 100
cumulative_sim = cumsum(phase_completions) ./ n_sims * 100

println("\nSimulation Results:")
println("  Average pulls: $(round(sim_mean, digits=2))")
println("\nCompletion by phase:")
println("Phase | Markov% | Simulation% | Difference")
println("------|---------|-------------|----------")
for phase in 1:6
    markov_pct = round(phase_results[phase].total_completion * 100, digits=2)
    sim_pct = round(phase_percentages[phase], digits=2)
    diff = round(abs(markov_pct - sim_pct), digits=2)
    println("  $(phase)   | $(lpad(markov_pct, 7))% | $(lpad(sim_pct, 11))% | $(lpad(diff, 8))%")
end

println("\nOverall Comparison:")
println("  Markov chain: $(round(total_expected, digits=2)) pulls")
println("  Simulation:   $(round(sim_mean, digits=2)) pulls") 
println("  Difference:   $(round(abs(total_expected - sim_mean), digits=2)) pulls")
println("  Error:        $(round(abs(total_expected - sim_mean) / sim_mean * 100, digits=3))%")