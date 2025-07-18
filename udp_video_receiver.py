import cv2
import socket
import struct
import threading
import time
import logging
import numpy as np
from collections import defaultdict, deque
from queue import Queue, Empty

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VideoReceiver:
    def __init__(self, host='127.0.0.1', port=12346, frame_timeout=1.0, 
                 video_width=640, video_height=480, video_channels=3,
                 packet_queue_size=2000, frame_queue_size=5):
        """
        Args:
            host: bound host address
            port: listening port
            frame_timeout: timeout in seconds for incomplete frames
            max_frame_fuffer: Maximum number of frames buffered
            videow_width: The width of a video frame
            video_ceight: The height of a video frame
            video_channels: The number of channels in a video frame (e.g. BGR is 3)
            packet_queue_size: The maximum size of the packet queue
            frame_queue_size: Maximum size of the complete frame queue
        """
        self.host = host
        self.port = port
        self.frame_timeout = frame_timeout
        
        # 视频尺寸信息
        self.video_width = video_width
        self.video_height = video_height
        self.video_channels = video_channels
        self.expected_frame_size = self.video_width * self.video_height * self.video_channels
        
        # 创建套接字
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        buffer_size = 2 * 1024 * 1024  # 2MB
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_size)
        logger.info(f"Socket 接收缓冲区大小已设置为 {buffer_size / 1024 / 1024} MB")
        self.socket.bind((host, port))
        self.socket.settimeout(0.1)  # 非阻塞并带超时

        # 线程间通信队列
        self.packet_queue = Queue(maxsize=packet_queue_size)  # 原始数据包队列
        self.frame_queue = Queue(maxsize=frame_queue_size)    # 完整帧队列
        
        # 帧重组缓冲区 (用于包解析线程)
        self.frame_buffers = defaultdict(dict)  # frame_id -> {packet_id: packet_data}
        self.frame_info = defaultdict(dict)     # frame_id -> {total_packets, received_packets, timestamp}
        
        # 线程锁
        self.frame_buffer_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # 控制标志
        self.is_running = False
        
        # 统计信息
        self.stats = {
            'packets_received': 0,
            'packets_parsed': 0,
            'packets_out_of_order': 0,
            'frames_completed': 0,
            'frames_dropped': 0,
            'frames_displayed': 0,
            'queue_packet_size': 0,
            'queue_frame_size': 0
        }

        logger.info(f"视频接收器已在 {host}:{port} 初始化")
        logger.info(f"等待 {video_width}x{video_height}x{video_channels} 的原始视频流")

    def update_stats(self, key, increment=1):
        """线程安全的统计信息更新"""
        with self.stats_lock:
            self.stats[key] += increment

    def get_stats(self):
        """获取统计信息的副本"""
        with self.stats_lock:
            return self.stats.copy()

    def packet_receiver_thread(self):
        """接收数据包并放入队列"""
        logger.info("数据包接收线程已启动")
        packet_count = 0
        
        while self.is_running:
            try:
                # 接收UDP数据包
                packet, addr = self.socket.recvfrom(2048)
                
                # 将原始数据包放入队列
                try:
                    self.packet_queue.put(packet, block=False)
                    self.update_stats('packets_received')
                    packet_count += 1
                except:
                    # 队列满了，丢弃最旧的包
                    try:
                        self.packet_queue.get_nowait()
                        self.packet_queue.put(packet, block=False)
                        logger.warning("数据包队列已满，丢弃旧包")
                    except:
                        pass
                        
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"接收数据包时出错: {e}")
            if packet_count % 1000 == 0:
                logger.debug(f"已接收 {packet_count} 个数据包")

    def parse_packet_header(self, packet):
        """解析数据包头部以提取元数据"""
        header_size = 16  # 头部大小 (字节)
        if len(packet) < header_size:
            return None
        
        header = packet[:header_size]
        data = packet[header_size:]

        try:
            frame_id, packet_id, total_packets, data_size = struct.unpack('!IIII', header)
            return {
                'frame_id': frame_id,
                'packet_id': packet_id,
                'total_packets': total_packets,
                'data_size': data_size,
                'data': data[:data_size]
            }
        except struct.error as e:
            logger.error(f"解析数据包头部时出错: {e}")
            return None

    def packet_parser_thread(self):
        """线程2: 解析数据包并重组帧"""
        logger.info("数据包解析线程已启动")
        
        while self.is_running:
            try:
                # 从队列中获取数据包
                packet = self.packet_queue.get(timeout=0.1)
                
                # 解析数据包头部
                packet_info = self.parse_packet_header(packet)
                if not packet_info:
                    continue
                
                self.update_stats('packets_parsed')
                
                frame_id = packet_info['frame_id']
                packet_id = packet_info['packet_id']
                total_packets = packet_info['total_packets']
                
                with self.frame_buffer_lock:
                    # 初始化帧信息
                    if frame_id not in self.frame_info:
                        self.frame_info[frame_id] = {
                            'total_packets': total_packets,
                            'received_packets': 0,
                            'first_packet_time': time.time()
                        }
                    
                    # 检查是否为重复包
                    if packet_id in self.frame_buffers[frame_id]:
                        logger.debug(f"收到重复数据包: 帧 {frame_id}, 包 {packet_id}")
                        continue
                    
                    # 存储包数据
                    self.frame_buffers[frame_id][packet_id] = packet_info['data']
                    self.frame_info[frame_id]['received_packets'] += 1
                    
                    # 检查包顺序
                    expected_next_packet = len(self.frame_buffers[frame_id]) - 1
                    if packet_id != expected_next_packet:
                        self.update_stats('packets_out_of_order')
                        logger.debug(f"乱序数据包: 帧 {frame_id}, 包 {packet_id}")
                    
                    # 检查帧是否完整
                    if self.frame_info[frame_id]['received_packets'] == total_packets:
                        self.complete_frame(frame_id)
                        
            except Empty:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"解析数据包时出错: {e}")

    def complete_frame(self, frame_id):
        """完成帧重组并添加到帧队列"""
        try:
            # 通过对数据包排序来重构帧数据
            frame_data_list = []
            for packet_id in sorted(self.frame_buffers[frame_id].keys()):
                frame_data_list.append(self.frame_buffers[frame_id][packet_id])
            frame_data = b''.join(frame_data_list)
            
            # 将完整帧放入队列
            frame_info = {
                'frame_id': frame_id,
                'data': frame_data,
                'timestamp': time.time()
            }
            
            try:
                self.frame_queue.put(frame_info, block=False)
                self.update_stats('frames_completed')
                logger.info(f"帧 {frame_id} 已完成并加入队列")
            except:
                # 队列满了，丢弃最旧的帧
                try:
                    dropped_frame = self.frame_queue.get_nowait()
                    self.frame_queue.put(frame_info, block=False)
                    self.update_stats('frames_dropped')
                    logger.warning(f"帧队列已满，丢弃帧 {dropped_frame['frame_id']}")
                except:
                    self.update_stats('frames_dropped')
                    logger.warning(f"无法加入帧队列，丢弃帧 {frame_id}")
            
            # 清理缓冲区
            del self.frame_buffers[frame_id]
            del self.frame_info[frame_id]
            
        except Exception as e:
            logger.error(f"完成帧 {frame_id} 时出错: {e}")

    def frame_display_thread(self):
        """线程3: 显示视频帧"""
        logger.info("帧显示线程已启动")
        
        while self.is_running:
            try:
                # 从队列中获取完整帧
                frame_info = self.frame_queue.get(timeout=0.1)
                
                # 显示帧
                if self.display_frame(frame_info['data']):
                    self.update_stats('frames_displayed')
                
                # 检查退出信号
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("收到退出信号")
                    self.is_running = False
                    break
                    
            except Empty:
                # 没有帧可显示，短暂休眠
                time.sleep(0.01)
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"显示帧时出错: {e}")

    def display_frame(self, frame_data):
        """解析原始数据并显示帧"""
        try:
            # 检查接收到的数据大小是否与预期的帧大小匹配
            if len(frame_data) != self.expected_frame_size:
                logger.error(f"帧数据大小不匹配! 预期: {self.expected_frame_size}, 收到: {len(frame_data)}")
                return False

            # 将字节数据转换为 NumPy 数组
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            
            # 将一维数组重塑为 (height, width, channels) 的图像格式
            frame = frame.reshape((self.video_height, self.video_width, self.video_channels))
            frame = frame.copy()
            
            if frame is not None:
                # 获取统计信息
                current_stats = self.get_stats()
                current_stats['queue_packet_size'] = self.packet_queue.qsize()
                current_stats['queue_frame_size'] = self.frame_queue.qsize()
                
                # 添加统计信息覆盖层
                stats_text = [
                    f"Packets Recv: {current_stats['packets_received']}",
                    f"Packets Parsed: {current_stats['packets_parsed']}",
                    f"Frames Complete: {current_stats['frames_completed']}",
                    f"Frames Displayed: {current_stats['frames_displayed']}",
                    f"Frames Dropped: {current_stats['frames_dropped']}",
                    f"Out-of-order: {current_stats['packets_out_of_order']}",
                    f"Packet Queue: {current_stats['queue_packet_size']}",
                    f"Frame Queue: {current_stats['queue_frame_size']}"
                ]
                
                y_offset = 20
                for i, text in enumerate(stats_text):
                    cv2.putText(frame, text, (10, y_offset + i * 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 显示帧
                cv2.imshow('Receiver - Video Stream', frame)
                return True
            else:
                logger.error("无法创建帧")
                return False
                
        except Exception as e:
            logger.error(f"显示帧时出错: {e}")
            return False

    def frame_cleanup_thread(self):
        """线程4: 清理过期帧"""
        logger.info("帧清理线程已启动")
        
        while self.is_running:
            self.cleanup_expired_frames()
            time.sleep(0.5)

    def cleanup_expired_frames(self):
        """移除过期的不完整帧"""
        current_time = time.time()
        expired_frames = []
        
        with self.frame_buffer_lock:
            # 查找过期帧
            for frame_id, info in list(self.frame_info.items()):
                if current_time - info['first_packet_time'] > self.frame_timeout:
                    expired_frames.append(frame_id)
            
            # 清理过期帧
            for frame_id in expired_frames:
                received = self.frame_info[frame_id]['received_packets']
                total = self.frame_info[frame_id]['total_packets']
                logger.warning(f"帧 {frame_id} 已过期 ({received}/{total} 个包)")
                
                del self.frame_buffers[frame_id]
                del self.frame_info[frame_id]
                self.update_stats('frames_dropped')

    def statistics_thread(self):
        """线程5: 统计信息报告"""
        logger.info("统计信息线程已启动")
        
        while self.is_running:
            time.sleep(5)
            stats = self.get_stats()
            stats['queue_packet_size'] = self.packet_queue.qsize()
            stats['queue_frame_size'] = self.frame_queue.qsize()
            
            logger.info(f"统计信息 - "
                       f"Packets recv/parsed: {stats['packets_received']}/{stats['packets_parsed']}, "
                       f"Frames complete/display/drop: {stats['frames_completed']}/{stats['frames_displayed']}/{stats['frames_dropped']}, "
                       f"Queue sizes P/F: {stats['queue_packet_size']}/{stats['queue_frame_size']}, "
                       f"Out-of-order: {stats['packets_out_of_order']}")

    def start_receiving(self):
        """开始视频接收"""
        self.is_running = True
        logger.info("开始接收视频...")
        
        # 创建所有线程
        threads = [
            threading.Thread(target=self.packet_receiver_thread, name="PacketReceiver", daemon=True),
            threading.Thread(target=self.packet_parser_thread, name="PacketParser", daemon=True),
            threading.Thread(target=self.frame_display_thread, name="FrameDisplay", daemon=True),
            threading.Thread(target=self.frame_cleanup_thread, name="FrameCleanup", daemon=True),
            threading.Thread(target=self.statistics_thread, name="Statistics", daemon=True)
        ]
        
        # 启动所有线程
        for thread in threads:
            thread.start()
            logger.info(f"线程 {thread.name} 已启动")
        
        try:
            # 只要负责UI的显示线程还在运行，主线程就保持存活
            # frame_display_thread 是列表中的第3个线程，索引为2
            display_thread = threads[2]
            while display_thread.is_alive():
                # 主线程可以短暂休眠，以降低CPU占用，同时保持对Ctrl+C的响应
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("用户中断了接收 (Ctrl+C)")
        finally:
            # 无论如何退出，都调用统一的停止流程
            self.stop_receiving()       

    def stop_receiving(self):
        """停止视频接收和清理"""
        if self.is_running:
            return
        
        # 短暂等待，让子线程有机会检测到标志并退出
        time.sleep(0.2) 
            
        # 清理资源
        cv2.destroyAllWindows()
        self.socket.close()
        
        # 清空队列
        try:
            while not self.packet_queue.empty():
                self.packet_queue.get_nowait()
        except:
            pass
            
        try:
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()
        except:
            pass
        
        # 输出最终统计信息
        final_stats = self.get_stats()
        logger.info("视频接收已停止")
        logger.info(f"最终统计 - "
                    f"Packets received: {final_stats['packets_received']}, "
                    f"Packets parsed: {final_stats['packets_parsed']}, "
                    f"Frames completed: {final_stats['frames_completed']}, "
                    f"Frames displayed: {final_stats['frames_displayed']}, "
                    f"Frames dropped: {final_stats['frames_dropped']}")

def main():
    """主函数，运行视频接收器"""
    try:
        # 配置视频的实际尺寸
        receiver = VideoReceiver(
            host='127.0.0.1',
            port=12345,
            frame_timeout=1.0,     # 帧过期时间
            video_width=640,       # 设置宽度
            video_height=480,      # 设置高度
            video_channels=3,      # 设置通道数 (BGR=3, 灰度=1)
            packet_queue_size=2000,# 数据包队列大小
            frame_queue_size=5      # 帧队列大小
        )
        receiver.start_receiving()
    except Exception as e:
        logger.error(f"main 函数出错: {e}")

if __name__ == "__main__":
    main()