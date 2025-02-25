import random
from typing import Dict

class CommunicationLossModel:
    """
    A packet loss model that creates bursts of packet loss with uniform length distribution.
    
    The model works as follows:
    1. There is a fixed probability of entering a loss period
    2. Once in a loss period, the length is uniformly sampled between min_loss_length and max_loss_length
    3. During a loss period, all packets are dropped
    4. After a loss period, there is a cooldown period during which no new loss periods can start
    """
    
    def __init__(self, 
                 params: Dict = None):
        """
        Initialize the Communication Loss Model.
        
        Args:
            params: Dictionary containing model parameters with keys:
                - loss_prob: Probability of entering a loss period (0.0-1.0)
                - min_loss_length: Minimum length of a loss period (packets)
                - max_loss_length: Maximum length of a loss period (packets)
                - cooldown_period: Number of packets to wait after a loss period before another can start
        """
        if params is None:
            params = {}
            
        self.loss_prob = params.get('loss_prob', 0.01)
        self.min_loss_length = params.get('min_loss_length', 1)
        self.max_loss_length = params.get('max_loss_length', 5)
        self.cooldown_period = params.get('cooldown_period', 10)
        
        # Internal state
        self.in_loss_period = False
        self.remaining_loss = 0
        self.cooldown_counter = 0
    
    def reset(self):
        """Reset the model state."""
        self.in_loss_period = False
        self.remaining_loss = 0
        self.cooldown_counter = 0
    
    def should_drop(self) -> bool:
        """
        Determine if a packet should be dropped.
        
        Args:
            seq_num: Sequence number of the packet
            
        Returns:
            True if the packet should be dropped, False otherwise
        """
        # If we're in a loss period, drop the packet
        if self.in_loss_period:
            self.remaining_loss -= 1
            
            # Check if we've reached the end of the loss period
            if self.remaining_loss <= 0:
                self.in_loss_period = False
                self.cooldown_counter = self.cooldown_period
            return True
        
        # If we're in cooldown, don't start a new loss period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        # Randomly decide if we should enter a loss period
        if random.random() < self.loss_prob:
            self.in_loss_period = True
            self.remaining_loss = random.randint(self.min_loss_length, self.max_loss_length)
            
            # Handle single packet loss case
            if self.remaining_loss <= 1:
                self.in_loss_period = False
                self.cooldown_counter = self.cooldown_period
            else:
                self.remaining_loss -= 1
            
            return True
        
        return False
