import socket
import threading
import sys
import os
import time
from datetime import datetime

class ChatServer:
    def __init__(self, host='0.0.0.0', port=4000):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = {}  # {socket: {'username': str, 'last_activity': float}}
        self.clients_lock = threading.Lock()
        self.running = False
        self.idle_timeout = 60  # seconds
        
    def start(self):
        """Start the chat server"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(10)
            self.running = True
            
            print(f"[SERVER] Chat server started on {self.host}:{self.port}")
            print(f"[SERVER] Waiting for connections...")
            
            # Start idle timeout checker thread
            timeout_thread = threading.Thread(target=self.check_idle_timeouts, daemon=True)
            timeout_thread.start()
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"[SERVER] New connection from {address}")
                    
                    # Start a new thread to handle this client
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        print(f"[SERVER ERROR] {e}")
                        
        except Exception as e:
            print(f"[SERVER ERROR] Failed to start server: {e}")
        finally:
            self.stop()
    
    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        username = None
        
        try:
            # Wait for LOGIN command
            client_socket.settimeout(30)  # 30 second timeout for login
            data = client_socket.recv(1024).decode('utf-8').strip()
            
            if not data.startswith('LOGIN '):
                client_socket.send(b'ERR invalid-command\n')
                client_socket.close()
                return
            
            username = data[6:].strip()
            
            # Validate username
            if not username:
                client_socket.send(b'ERR empty-username\n')
                client_socket.close()
                return
            
            # Check if username is already taken
            with self.clients_lock:
                for client_info in self.clients.values():
                    if client_info['username'] == username:
                        client_socket.send(b'ERR username-taken\n')
                        client_socket.close()
                        return
                
                # Add client to active users
                self.clients[client_socket] = {
                    'username': username,
                    'last_activity': time.time()
                }
            
            # Send OK response
            client_socket.send(b'OK\n')
            print(f"[SERVER] User '{username}' logged in from {address}")
            
            # Remove timeout for regular communication
            client_socket.settimeout(None)
            
            # Main message loop
            buffer = ""
            while self.running:
                try:
                    data = client_socket.recv(1024).decode('utf-8')
                    
                    if not data:
                        break
                    
                    # Update last activity
                    with self.clients_lock:
                        if client_socket in self.clients:
                            self.clients[client_socket]['last_activity'] = time.time()
                    
                    buffer += data
                    
                    # Process complete lines
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        self.process_command(client_socket, username, line)
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[SERVER ERROR] Error receiving from {username}: {e}")
                    break
                    
        except Exception as e:
            print(f"[SERVER ERROR] Error handling client {address}: {e}")
        finally:
            self.disconnect_client(client_socket, username)
    
    def process_command(self, client_socket, username, command):
        """Process client commands"""
        try:
            if command.startswith('MSG '):
                # Broadcast message
                message = command[4:].strip()
                if message:
                    self.broadcast(f"MSG {username} {message}\n", exclude=client_socket)
                    
            elif command.startswith('DM '):
                # Private message
                parts = command[3:].split(' ', 1)
                if len(parts) < 2:
                    client_socket.send(b'ERR invalid-dm-format\n')
                    return
                
                target_username = parts[0].strip()
                message = parts[1].strip()
                
                if not message:
                    client_socket.send(b'ERR empty-message\n')
                    return
                
                self.send_private_message(username, target_username, message, client_socket)
                
            elif command == 'WHO':
                # List active users
                self.list_users(client_socket)
                
            elif command == 'PING':
                # Heartbeat response
                client_socket.send(b'PONG\n')
                
            else:
                client_socket.send(b'ERR unknown-command\n')
                
        except Exception as e:
            print(f"[SERVER ERROR] Error processing command from {username}: {e}")
    
    def broadcast(self, message, exclude=None):
        """Broadcast message to all connected clients except the excluded one"""
        with self.clients_lock:
            disconnected = []
            for client_socket in self.clients.keys():
                if client_socket != exclude:
                    try:
                        client_socket.send(message.encode('utf-8'))
                    except:
                        disconnected.append(client_socket)
            
            # Remove disconnected clients
            for client_socket in disconnected:
                username = self.clients[client_socket]['username']
                del self.clients[client_socket]
                print(f"[SERVER] Removed disconnected user: {username}")
    
    def send_private_message(self, from_username, to_username, message, sender_socket):
        """Send private message to specific user"""
        with self.clients_lock:
            target_socket = None
            for client_socket, client_info in self.clients.items():
                if client_info['username'] == to_username:
                    target_socket = client_socket
                    break
            
            if target_socket:
                try:
                    target_socket.send(f"DM {from_username} {message}\n".encode('utf-8'))
                    sender_socket.send(f"DM-SENT {to_username}\n".encode('utf-8'))
                except:
                    sender_socket.send(b'ERR dm-failed\n')
            else:
                sender_socket.send(b'ERR user-not-found\n')
    
    def list_users(self, client_socket):
        """Send list of active users to client"""
        with self.clients_lock:
            for client_info in self.clients.values():
                try:
                    client_socket.send(f"USER {client_info['username']}\n".encode('utf-8'))
                except:
                    pass
    
    def check_idle_timeouts(self):
        """Check for idle users and disconnect them"""
        while self.running:
            time.sleep(10)  # Check every 10 seconds
            
            with self.clients_lock:
                current_time = time.time()
                disconnected = []
                
                for client_socket, client_info in self.clients.items():
                    if current_time - client_info['last_activity'] > self.idle_timeout:
                        disconnected.append((client_socket, client_info['username']))
                
                for client_socket, username in disconnected:
                    print(f"[SERVER] User '{username}' timed out due to inactivity")
                    try:
                        client_socket.send(b'INFO timeout due to inactivity\n')
                        client_socket.close()
                    except:
                        pass
                    
                    if client_socket in self.clients:
                        del self.clients[client_socket]
                    
                    # Notify other users
                    self.broadcast(f"INFO {username} disconnected\n")
    
    def disconnect_client(self, client_socket, username):
        """Handle client disconnection"""
        with self.clients_lock:
            if client_socket in self.clients:
                del self.clients[client_socket]
        
        try:
            client_socket.close()
        except:
            pass
        
        if username:
            print(f"[SERVER] User '{username}' disconnected")
            self.broadcast(f"INFO {username} disconnected\n")
    
    def stop(self):
        """Stop the server"""
        print("[SERVER] Shutting down...")
        self.running = False
        
        # Close all client connections
        with self.clients_lock:
            for client_socket in list(self.clients.keys()):
                try:
                    client_socket.close()
                except:
                    pass
            self.clients.clear()
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("[SERVER] Server stopped")


def main():
    # Get port from environment variable or command line argument
    port = int(os.getenv('CHAT_PORT', '4000'))
    
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Usage: python chat_server.py [port]")
            sys.exit(1)
    
    # Create and start server
    server = ChatServer(port=port)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Received shutdown signal")
        server.stop()


if __name__ == '__main__':
    main()