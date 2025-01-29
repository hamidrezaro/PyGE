import numpy as np
import random
from scipy.stats import expon
from scipy.optimize import minimize_scalar

def pareto_type_ii_sample(alpha, lambda_param, size=1):
    u = np.random.uniform(0, 1, size)  # Uniform random values
    return lambda_param * ((1 - u) ** (-1 / alpha) - 1)

class AdvancedPacketLossModel:
    def __init__(self, state_params):
        """
        Initialize the HMM with state-specific parameters.
        :param state_params: Dictionary with states and their corresponding probability distribution parameters.
        """
        self.state_params = state_params  # Dict of state-specific params
        self.current_state = 'Good'      # Start in Good state

    def transition_state(self):
        """
        Transition between states based on predefined probabilities.
        """
        transition_probs = self.state_params[self.current_state]['transitions']
        self.current_state = random.choices(
            list(transition_probs.keys()), 
            weights=list(transition_probs.values())
        )[0]

    def sample_bll(self):
        """
        Sample Burst Loss Length (BLL) for the current state.
        """
        dist = self.state_params[self.current_state]['distribution']
        params = self.state_params[self.current_state]['params']

        if dist == 'pareto':
            # Use the Pareto Type II distribution for sampling
            return int(pareto_type_ii_sample(params['alpha'], params['lambda'], size=1)[0])
        elif dist == 'exponential':
            # Use the exponential distribution for sampling
            return int(expon.rvs(scale=params['mu']))
        else:
            raise ValueError("Unsupported distribution type.")

    def process_packet(self, bll):
        """
        Process a packet: Determine if it should be dropped or passed.
        :param bll: Burst Loss Length to process
        """
        for _ in range(bll):
            self.transition_state()  # Evaluate state transitions during the burst
            yield True  # Drop the packet
        yield False  # Allow one successful packet after the burst

    

# Updated Example Usage with two states
import json
with open('src/pyge/config.json', 'r') as file:
    state_params = json.load(file)["pareto_burst_length_distribution"]


# Instantiate the model
model = AdvancedPacketLossModel(state_params)

# Simulate packet processing
total_packets = 100
dropped_packets = 0
total_bll = 0
bll_count = 0

packet = 0
while packet < total_packets:
    bll = model.sample_bll()  # Sample BLL
    if bll > 0:
        total_bll += bll
        bll_count += 1
        # Drop bll consecutive packets
        for _ in range(min(bll, total_packets - packet)):
            print(f"Packet {packet} dropped")
            dropped_packets += 1
            packet += 1
        if packet < total_packets:
            # Deliver one packet after burst
            print(f"Packet {packet} delivered")
            packet += 1
    else:
        # No burst, deliver packet
        print(f"Packet {packet} delivered")
        packet += 1

loss_rate = dropped_packets / total_packets
avg_bll = total_bll / bll_count if bll_count > 0 else 0
print(f"\nPacket Loss Rate: {loss_rate:.2%}")
print(f"Average Burst Loss Length: {avg_bll:.2f}")


