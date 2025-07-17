import socket
import threading
import time
import random
import logging
from queue import Queue, Empty

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChannelSimulator:
    def __init__(self, sender_port=12345, receiver_port=12346, 
                 packet_loss_rate=0.05, max_delay=0.1, reorder_probability=0.1):
        """
        Initialize channel simulator
        
        Args:
            sender_port: Port to receive packets from sender
            receiver_port: Port to forward packets to receiver
            packet_loss_rate: Probability of packet loss (0.0 - 1.0)
            max_delay: Maximum delay in seconds
            reorder_probability: Probability of packet reordering (0.0 - 1.0)
        """
        self.sender_port = sender_port
        self.receiver_port = receiver_port
        self.packet_loss_rate = packet_loss_rate
        self.max_delay = max_delay
        self.reorder_probability = reorder_probability
        
        # Create sockets
        self.receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sender_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # 设置一个1.0秒的超时。这意味着recvfrom最多只会阻塞1秒。
        self.receiver_socket.settimeout(1.0)
        
        # Bind receiver socket
        self.receiver_socket.bind(('127.0.0.1', sender_port))
        
        # Packet buffer for reordering
        self.packet_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'packets_received': 0,
            'packets_dropped': 0,
            'packets_delayed': 0,
            'packets_reordered': 0,
            'packets_forwarded': 0
        }
        
        self.is_running = False
        logger.info(f"Channel simulator initialized: {sender_port} -> {receiver_port}")
        logger.info(f"Loss rate: {packet_loss_rate*100:.1f}%, Max delay: {max_delay*1000:.1f}ms")
    
    def simulate_packet_loss(self):
        """Simulate packet loss"""
        return random.random() < self.packet_loss_rate
    
    def simulate_delay(self):
        """Generate random delay"""
        return random.uniform(0, self.max_delay)
    
    def should_reorder(self):
        """Determine if packet should be reordered"""
        return random.random() < self.reorder_probability
    
    def forward_packet(self, packet, addr, delay=0):
        """Forward packet to receiver with optional delay"""
        if delay > 0:
            time.sleep(delay)
        
        try:
            self.sender_socket.sendto(packet, ('127.0.0.1', self.receiver_port))
            self.stats['packets_forwarded'] += 1
            logger.debug(f"Forwarded packet (delay: {delay*1000:.1f}ms)")
        except Exception as e:
            logger.error(f"Error forwarding packet: {e}")
    
    def process_packet_buffer(self):
        """Process packets in buffer for reordering"""
        while self.is_running:
            try:
                with self.buffer_lock:
                    if self.packet_buffer:
                        # Sort by timestamp and process older packets
                        self.packet_buffer.sort(key=lambda x: x[2])
                        current_time = time.time()
                        
                        # Process packets older than 50ms
                        to_remove = []
                        for i, (packet, addr, timestamp) in enumerate(self.packet_buffer):
                            if current_time - timestamp > 0.05:  # 50ms threshold
                                threading.Thread(
                                    target=self.forward_packet,
                                    args=(packet, addr, 0),
                                    daemon=True
                                ).start()
                                to_remove.append(i)
                        
                        # Remove processed packets
                        for i in reversed(to_remove):
                            self.packet_buffer.pop(i)
                
                time.sleep(0.01)  # Check every 10ms
                
            except Exception as e:
                logger.error(f"Error processing packet buffer: {e}")
    
    def handle_packet(self, packet, addr):
        """Handle incoming packet with channel simulation"""
        self.stats['packets_received'] += 1
        
        # Simulate packet loss
        if self.simulate_packet_loss():
            self.stats['packets_dropped'] += 1
            logger.debug(f"Packet dropped (loss simulation)")
            return
        
        # Simulate delay
        delay = self.simulate_delay()
        if delay > 0:
            self.stats['packets_delayed'] += 1
        
        # Simulate reordering
        if self.should_reorder() and delay > 0.02:  # Only reorder if delay > 20ms
            self.stats['packets_reordered'] += 1
            with self.buffer_lock:
                self.packet_buffer.append((packet, addr, time.time()))
            logger.debug(f"Packet buffered for reordering")
        else:
            # Forward packet immediately with delay
            if delay > 0:
                threading.Thread(
                    target=self.forward_packet,
                    args=(packet, addr, delay),
                    daemon=True
                ).start()
            else:
                self.forward_packet(packet, addr)
    
    def start_simulation(self):
        """Start channel simulation"""
        self.is_running = True
        logger.info("Starting channel simulation...")
        
        # Start packet buffer processing thread
        buffer_thread = threading.Thread(target=self.process_packet_buffer, daemon=True)
        buffer_thread.start()
        
        # Start statistics reporting thread
        stats_thread = threading.Thread(target=self.report_statistics, daemon=True)
        stats_thread.start()
        
        try:
            while self.is_running:
                try:
                    # Receive packet from sender
                    packet, addr = self.receiver_socket.recvfrom(2048)
                    
                    # Handle packet in separate thread for better performance
                    threading.Thread(
                        target=self.handle_packet,
                        args=(packet, addr),
                        daemon=True
                    ).start()
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error receiving packet: {e}")
                    
        except KeyboardInterrupt:
            logger.info("Channel simulation interrupted by user")
        finally:
            self.stop_simulation()
    
    def report_statistics(self):
        """Report channel statistics periodically"""
        while self.is_running:
            time.sleep(5)  # Report every 5 seconds
            if self.stats['packets_received'] > 0:
                loss_rate = (self.stats['packets_dropped'] / self.stats['packets_received']) * 100
                reorder_rate = (self.stats['packets_reordered'] / self.stats['packets_received']) * 100
                delay_rate = (self.stats['packets_delayed'] / self.stats['packets_received']) * 100
                
                logger.info(f"Channel Stats - Received: {self.stats['packets_received']}, "
                           f"Dropped: {self.stats['packets_dropped']} ({loss_rate:.1f}%), "
                           f"Reordered: {self.stats['packets_reordered']} ({reorder_rate:.1f}%), "
                           f"Delayed: {self.stats['packets_delayed']} ({delay_rate:.1f}%)")
    
    def stop_simulation(self):
        """Stop channel simulation and cleanup"""
        self.is_running = False
        self.receiver_socket.close()
        self.sender_socket.close()
        logger.info("Channel simulation stopped")
        
        # Final statistics
        if self.stats['packets_received'] > 0:
            loss_rate = (self.stats['packets_dropped'] / self.stats['packets_received']) * 100
            logger.info(f"Final Stats - Total received: {self.stats['packets_received']}, "
                       f"Total dropped: {self.stats['packets_dropped']} ({loss_rate:.1f}%), "
                       f"Total forwarded: {self.stats['packets_forwarded']}")

def main():
    """Main function to run the channel simulator"""
    try:
        # Configure channel parameters
        channel = ChannelSimulator(
            sender_port=12345,
            receiver_port=12346,
            packet_loss_rate=0.001,    # 2% packet loss
            max_delay=0.005,           # Max 10ms delay
            reorder_probability=0.01  # 1% reordering
        )
        channel.start_simulation()
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
