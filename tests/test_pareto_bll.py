from pyge.models import GEParetoBLLModel

import json
with open('src/pyge/config.json', 'r') as file:
    state_params = json.load(file)["pareto_burst_length_distribution"]


def simulate_pareto_bll(state_params, total_packets=100):
    # Instantiate the model
    model = GEParetoBLLModel(state_params)

    # Simulate packet processing
    dropped_packets = 0
    total_bll = 0
    bll_count = 0
    prev_dropped = False  # Track previous packet state

    for packet in range(total_packets):
        current_dropped = model.process_packet()
        
        if current_dropped:
            dropped_packets += 1
            
            # Capture new burst length at start of burst
            if not prev_dropped:
                # Add 1 to account for decrement that just occurred in process_packet
                burst_length = model.bll + 1
                total_bll += burst_length
                bll_count += 1
        
        prev_dropped = current_dropped  # Update state tracking

    loss_rate = dropped_packets / total_packets
    avg_bll = total_bll / bll_count if bll_count > 0 else 0
    
    return loss_rate, avg_bll

loss_rate, avg_bll = simulate_pareto_bll(state_params)

print(f"\nPacket Loss Rate: {loss_rate:.2%}")
print(f"Average Burst Loss Length: {avg_bll:.2f}")
