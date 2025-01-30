import socket
import threading
import time
from scapy.all import *
from scapy.config import conf
from scapy.layers.inet import IP, TCP, UDP
from pyge.models import GEClassicModel, GEParetoBLLModel
import signal
import sys
import struct
from collections import deque
from threading import Lock, Event
import lz4.frame

class NetworkEmulator:
    def __init__(self, input_port: int, output_port: int, 
                 model_name: str, params_path: str, 
                 protocol: str = 'udp', output_ip: str = '127.0.0.1',
                 log_packets: bool = False, log_path: str = "packet_log.bin"):
        self.input_port = input_port
        self.output_port = output_port
        self.output_ip = output_ip
        self.protocol = protocol.lower()
        self.running = False
        self.loss_model = self._init_loss_model(model_name, params_path)
        
        # TCP specific variables
        self.tcp_connections = {}  # Track sequence numbers and ACKs
        self.tcp_lock = threading.Lock()

        self.log_packets = log_packets
        self.log_path = log_path
        self.log_queue = deque()
        self.log_lock = Lock()
        self.log_event = Event()
        self.log_thread = None

    def _init_loss_model(self, model_name: str, params_path: str):
        """Initialize the packet loss model from config file"""
        with open(params_path) as f:
            params = json.load(f)
            
        if model_name.lower() == 'GE_Classic':
            params = params['geclassic']
            return GEClassicModel(params)
        elif model_name.lower() == 'GE_Pareto_BLL':
            params = params['geparetobll']
            return GEParetoBLLModel(params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _init_logger(self):
        """Initialize packet logging system"""
        print(f"[Logger] Starting packet writer thread for {self.log_path}")
        self.log_thread = threading.Thread(target=self._packet_writer)
        self.log_thread.daemon = True
        self.log_thread.start()

    def _log_packet(self, data: bytes, dropped: bool):
        """Add packet to log queue (thread-safe)"""
        if not self.log_packets:
            return
            
        timestamp = time.time_ns()
        entry = (timestamp, len(data), dropped, data)
        
        with self.log_lock:
            self.log_queue.append(entry)
            self.log_event.set()
            # Debug print every 100 packets
            if len(self.log_queue) % 100 == 0:
                print(f"[Logger] Queue size: {len(self.log_queue)} packets")

    def _packet_writer(self):
        """Dedicated thread for writing packets to disk"""
        BUFFER_SIZE = 1000
        buffer = []
        total_written = 0
        print(f"[Writer] Starting with buffer size {BUFFER_SIZE}")
        
        try:
            with open(self.log_path, "ab") as f, lz4.frame.LZ4FrameFile(f, 'wb') as compressed_file:
                print(f"[Writer] Opened log file: {self.log_path}")
                
                while self.running or buffer:
                    self.log_event.wait(timeout=1)
                    
                    # Collect data from queue
                    with self.log_lock:
                        qsize = len(self.log_queue)
                        if qsize > 0:
                            while self.log_queue:
                                buffer.append(self.log_queue.popleft())
                            self.log_event.clear()

                    # Write when buffer reaches threshold
                    if len(buffer) >= BUFFER_SIZE or not self.running:
                        write_time = time.time()
                        batch_size = len(buffer)
                        binary_data = b''.join(
                            struct.pack(
                                "!QH?%ds" % length,
                                ts, length, dropped, data
                            ) for ts, length, dropped, data in buffer
                        )
                        
                        compressed_file.write(binary_data)
                        compressed_file.flush()
                        total_written += batch_size
                        print(f"[Writer] Wrote {batch_size} packets ({len(binary_data)//1024}KB) "
                              f"in {(time.time()-write_time)*1000:.1f}ms "
                              f"(Total: {total_written})")
                        buffer.clear()
                        
                print(f"[Writer] Final flush completed. Total packets written: {total_written}")
                
        except Exception as e:
            print(f"[Writer] Critical error: {str(e)}")
            raise

    def _udp_listener(self):
        """Handle UDP packet loss simulation"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', self.input_port))
        
        forward_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        while self.running:
            data, addr = sock.recvfrom(4096)
            drop = self.loss_model.process_packet()
            
            if self.log_packets:
                self._log_packet(data, drop)
                
            if not drop:
                forward_sock.sendto(data, (self.output_ip, self.output_port))
        
        forward_sock.close()
        sock.close()

    def _tcp_packet_handler(self, pkt):
        """Handle TCP packet loss and ACK manipulation using Scapy"""
        if IP in pkt and TCP in pkt:
            with self.tcp_lock:
                # Track connection state
                conn_id = (pkt[IP].src, pkt[IP].dst, pkt[TCP].sport, pkt[TCP].dport)
                reverse_id = (pkt[IP].dst, pkt[IP].src, pkt[TCP].dport, pkt[TCP].sport)

                # Handle SYN packets
                if pkt[TCP].flags & 0x02:
                    self.tcp_connections[conn_id] = {'seq': pkt[TCP].seq, 'ack': 0}
                    return  # Don't forward SYN, let normal handshake happen
                
                drop = self.loss_model.process_packet()
                # Check if we should drop this packet
                if drop:
                    # Don't forward packet and don't send ACK
                    del self.tcp_connections[conn_id]
                    return

                # Forward packet and track sequence numbers
                if conn_id in self.tcp_connections:
                    # Modify sequence numbers to maintain connection
                    prev_seq = self.tcp_connections[conn_id]['seq']
                    new_seq = prev_seq + len(pkt[TCP].payload)
                    self.tcp_connections[conn_id]['seq'] = new_seq

                    # Create modified packet
                    new_pkt = IP(dst=self.output_ip)/TCP(
                        dport=self.output_port,
                        sport=pkt[TCP].sport,
                        seq=prev_seq,
                        ack=pkt[TCP].ack
                    )/pkt[TCP].payload
                    
                    send(new_pkt, verbose=0)
                    
                    # Update reverse connection tracking
                    self.tcp_connections[reverse_id] = {
                        'seq': pkt[TCP].ack,
                        'ack': new_seq + 1
                    }

                # Log original packet
                if self.log_packets:
                    raw_data = bytes(pkt[TCP].payload)
                    self._log_packet(raw_data, drop)

    def start(self):
        """Start the network emulator"""
        self.running = True
        if self.log_packets:
            self._init_logger()
        
        if self.protocol == 'udp':
            self.thread = threading.Thread(target=self._udp_listener)
            self.thread.daemon = True  # Make thread a daemon
            self.thread.start()
        elif self.protocol == 'tcp':
            # Start Scapy sniffer in separate thread
            self.thread = threading.Thread(
                target=lambda: sniff(
                    filter=f"tcp and port {self.input_port}",
                    prn=self._tcp_packet_handler,
                    store=0
                )
            )
            self.thread.daemon = True  # Make thread a daemon
            self.thread.start()
        else:
            raise ValueError("Invalid protocol. Use 'tcp' or 'udp'")

    def stop(self):
        """Stop the network emulator"""
        self.running = False
        if self.protocol == 'tcp':
            send(IP(dst="127.0.0.1")/TCP(dport=self.input_port), verbose=0)
        if self.thread:
            print("[Network] Stopping logger thread...")
            self.log_event.set()
            if self.log_thread:
                self.log_thread.join(timeout=2)
                if self.log_thread.is_alive():
                    print("[Network] WARNING: Logger thread did not terminate cleanly")
                else:
                    print("[Network] Logger thread stopped successfully")
            self.thread.join(timeout=1)
        self.tcp_connections.clear()

    def _signal_handler(self, signum, frame):
        """Handle termination signals"""
        self.stop()
        sys.exit(0)

# Example usage:
if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Network packet loss emulator')
    parser.add_argument('--input-port', type=int, required=True,
                      help='Port to listen for incoming packets')
    parser.add_argument('--output-port', type=int, required=True, 
                      help='Port to forward packets to')
    parser.add_argument('--output-ip', default='127.0.0.1',
                      help='IP address to forward packets to (default: 127.0.0.1)')
    parser.add_argument('--protocol', choices=['tcp', 'udp'], required=True,
                      help='Network protocol (tcp or udp)')
    parser.add_argument('--model', choices=['GE_Classic', 'GE_Pareto_BLL'], required=True,
                      help='Packet loss model to use')
    parser.add_argument('--config', required=True,
                      help='Path to model parameters config file')
    parser.add_argument('--log', help='Path to packet log file (optional)')

    args = parser.parse_args()

    # Set up signal handler
    def signal_handler(signum, frame):
        emulator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create and start emulator
    emulator = NetworkEmulator(
        input_port=args.input_port,
        output_port=args.output_port,
        output_ip=args.output_ip,
        model_name=args.model,
        params_path=args.config,
        protocol=args.protocol,
        log_packets=bool(args.log),
        log_path=args.log if args.log else "packet_log.bin"
    )

    try:
        emulator.start()
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
        signal_handler(None, None)
