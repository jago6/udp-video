import cv2
import socket
import struct
import threading
import time
import logging
from queue import Queue, Empty, Full 
import numpy as np

# (日志配置部分不变)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiThreadedVideoSender:

    def __init__(self, host='127.0.0.1', port=12345, max_packet_size=1024, 
                 packets_per_burst=50, burst_sleep_time=0.001, width=640, height=480, fps=30,
                 frame_queue_size=5, packet_queue_size=2000):
        self.host = host
        self.port = port
        self.max_packet_size = max_packet_size
        self.packets_per_burst = packets_per_burst
        self.burst_sleep_time = burst_sleep_time
        self.width = width
        self.height = height
        self.fps = fps
        
        # Thread control
        self.is_running = False
        self.threads = []
        
        # Sequence counters (thread-safe)
        self.frame_sequence = 0
        self.packet_sequence = 0
        self.sequence_lock = threading.Lock()
        
        # Queues for inter-thread communication
        self.frame_queue = Queue(maxsize=frame_queue_size)   # Original frame queue
        self.packet_queue = Queue(maxsize=packet_queue_size) # Packet queue
        
        # Initialize socket
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Initialize camera
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            logger.error("Failed to open camera")
            raise RuntimeError("Camera not available")
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Set the buffer size to 1 to reduce latency
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        logger.info(f"Multi threaded video transmitter initialization completed {host}:{port}")

    def get_next_frame_id(self):
        """Thread-safe frame ID generator"""
        with self.sequence_lock:
            frame_id = self.frame_sequence
            self.frame_sequence += 1
            return frame_id

    def create_packet_header(self, frame_id, packet_id, total_packets, data_size):
        """Create packet header with frame ID, packet ID, total packets, and data size"""
        return struct.pack('!IIII', frame_id, packet_id, total_packets, data_size)

    def split_frame_to_packets(self, frame_data, frame_id):
        """Split frame data into multiple UDP packets"""
        packets = []
        header_size = 16 # Size of the header in bytes
        data_per_packet = self.max_packet_size - header_size
        total_packets = (len(frame_data) + data_per_packet - 1) // data_per_packet
        
        for i in range(total_packets):
            start_idx = i * data_per_packet
            end_idx = min((i + 1) * data_per_packet, len(frame_data))
            packet_data = frame_data[start_idx:end_idx]
            
            header = self.create_packet_header(frame_id, i, total_packets, len(packet_data))
            packet = header + packet_data
            packets.append(packet)
            
        return packets

    
    def camera_display_thread(self):
        '''Camera acquisition and display thread - responsible for real-time acquisition of frame data and display'''
        logger.info("Camera display thread startup")
        
        frame_interval = 1.0 / self.fps  # Target frame rate
        while self.is_running:
            start_time = time.time()
            try:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to read camera frame")
                    time.sleep(0.1)
                    continue
                
                # Display local video
                # waste much time on cv2.imshow, so comment it out
                # cv2.imshow('Sender - Local Video', frame)
                
                # Check exit conditions
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Received exit signal")
                    self.is_running = False 
                    break
                
                # get frame ID
                frame_id = self.get_next_frame_id()
                
                try:
                    # Use non blocking put_nowait, discard if full
                    self.frame_queue.put_nowait((frame_id, frame))
                except Full:
                    logger.warning("Frame queue is full, discard one frame to maintain real-time performance")
      
                    pass
                
                # Dynamic frame rate control
               
                elapsed = time.time() - start_time
                wait_time = frame_interval - elapsed
                
                # logger.debug(f"now processing frame {frame_id},wait time: {wait_time:.4f} seconds")
                if wait_time > 0:
                    time.sleep(wait_time)

            except Exception as e:
                logger.error(f"Camera display thread error: {e}")
                self.is_running = False 
                
        logger.info("Camera displays thread end")

    def packet_processing_thread(self):
        '''Packet processing thread - responsible for converting frame data into data packets'''
        
        logger.info("Packet processing thread startup")
        while self.is_running or not self.frame_queue.empty():

            try:
                # Retrieve frame data from the frame queue
                frame_id, frame = self.frame_queue.get(timeout=0.001)
                
                start_time = time.time()
                
                # Convert frames into byte data
                frame_data = frame.tobytes()
                
                # Split into data packets
                packets = self.split_frame_to_packets(frame_data, frame_id)
                
                # Put the data packet into the sending queue
                for packet in packets:
                    self.packet_queue.put(packet, timeout=0.001)
                    
                elapsed = time.time() - start_time
                logger.debug(f"帧 {frame_id} 处理完成 ({len(packets)} 个包), 耗时: {elapsed:.4f}秒")

            except Empty:
                if not self.is_running: break
                continue
            except Full:
                logger.warning("Packet queue is full, processing thread paused")
                #time.sleep(0.01) 
            except Exception as e:
                logger.error(f"Packet processing thread error: {e}")
                self.is_running = False
        logger.info("Packet processing thread end")
        
    def network_sending_thread(self):
        '''Network sending thread - responsible for real-time sending of data packets'''
        
        logger.info("Network sending thread startup")
        
        packet_count = 0
        
        start_time = time.time()
        while self.is_running or not self.packet_queue.empty():
            try:
                packet = self.packet_queue.get(timeout=0.001)
                self.socket.sendto(packet, (self.host, self.port))
                packet_count += 1
                if packet_count % self.packets_per_burst == 0:
                    time.sleep(0)
                    
                if packet_count % 1000 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"已发送 {packet_count} 个数据包，耗时: {elapsed:.4f} 秒，平均速率: {1000 / elapsed:.2f} 包/秒")
                    start_time = time.time()  # 重置计时器
            except Empty:
                if not self.is_running: break
                continue
            except Exception as e:
                logger.error(f"Network sending thread error: {e}")
                self.is_running = False
        
        logger.info("Network sending thread end")
        
    def stats_thread(self):
        """Statistics thread - responsible for monitoring and reporting queue sizes"""
        
        logger.info("Statistics thread startup")
        while self.is_running:
            time.sleep(2)
            frame_queue_size = self.frame_queue.qsize()
            packet_queue_size = self.packet_queue.qsize()
            
            logger.info(f"Queue Status - Frame queue: {frame_queue_size}, Packet queue: {packet_queue_size}")
            
        logger.info("Statistics thread end")


    def start_streaming(self):
        """Start multi-threaded video streaming"""
        self.is_running = True
        logger.info("启动多线程视频流...")
        
        threads_config = [
            ("摄像头显示线程", self.camera_display_thread),
            ("数据包处理线程", self.packet_processing_thread),
            ("网络发送线程", self.network_sending_thread),
            ("统计线程", self.stats_thread)
        ]
        
        for name, target in threads_config:
            thread = threading.Thread(target=target, name=name)
            # daemon 设置为 True, 这样主线程退出时它们也会被强制退出
            thread.daemon = True 
            thread.start()
            self.threads.append(thread)
            logger.info(f"{name} 已启动")
        
        # 主线程进入一个响应式循环，而不是阻塞的 join
        try:
            # 只要负责UI的线程还在运行，主线程就保持存活
            # camera_thread 是 self.threads[0]
            while self.threads[0].is_alive():
                # 主线程可以做一些轻量级的工作，或者只是休眠
                # 这样可以保持对 Ctrl+C 的响应
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal (Ctrl+C)")
        finally:

            self.stop_streaming()

    def stop_streaming(self):
        """Stop video streaming and clean up resources"""
        if not self.is_running:

            return
            
        logger.info("Stopping video streaming ...")
        self.is_running = False
        
        # Waiting for all non daemon threads to end
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1) 
        
        # Clean up resources
        if self.camera.isOpened():
            self.camera.release()
        cv2.destroyAllWindows()
        self.socket.close()
        
        logger.info("Video stream has stopped, resources have been cleared")

def main():
    '''main'''
    try:
        sender = MultiThreadedVideoSender(
            host='127.0.0.1', 
            port=12345, 
            max_packet_size=1024,
            packets_per_burst=100,
            burst_sleep_time=0.00,
            width=640,
            height=480,
            fps=60,
            frame_queue_size=5,
            packet_queue_size=2000
        )
        sender.start_streaming()
    except Exception as e:
        logger.error(f"Main function error: {e}")

if __name__ == "__main__":
    main()