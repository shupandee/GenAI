import socket
import threading
import sys

class ChatClient:
    def __init__(self, host='localhost', port=4000):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
    def connect(self):
        """Connect to the chat server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.running = True
            print(f"Connected to chat server at {self.host}:{self.port}")
            print("=" * 50)
            return True
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
    
    def receive_messages(self):
        """Receive messages from server"""
        while self.running:
            try:
                message = self.socket.recv(1024).decode('utf-8')
                if message:
                    print(message, end='')
                else:
                    print("\n[Disconnected from server]")
                    self.running = False
                    break
            except Exception as e:
                if self.running:
                    print(f"\n[Error receiving message: {e}]")
                break
    
    def send_messages(self):
        """Send messages to server"""
        print("\nCommands:")
        print("  LOGIN <username>  - Log in with a username")
        print("  MSG <text>        - Send a message to everyone")
        print("  DM <user> <text>  - Send private message")
        print("  WHO               - List active users")
        print("  PING              - Test connection")
        print("  quit              - Exit chat")
        print("=" * 50)
        
        while self.running:
            try:
                message = input()
                
                if message.lower() == 'quit':
                    print("Goodbye!")
                    self.running = False
                    break
                
                if message.strip():
                    self.socket.send((message + '\n').encode('utf-8'))
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"[Error sending message: {e}]")
                break
    
    def start(self):
        """Start the chat client"""
        if not self.connect():
            return
        
        # Start receiving thread
        receive_thread = threading.Thread(target=self.receive_messages, daemon=True)
        receive_thread.start()
        
        # Start sending (main thread)
        self.send_messages()
        
        # Cleanup
        self.close()
    
    def close(self):
        """Close connection"""
        self.running = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass


def main():
    host = 'localhost'
    port = 4000
    
    # Allow custom host and port
    if len(sys.argv) > 1:
        host = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    client = ChatClient(host, port)
    client.start()


if __name__ == '__main__':
    main()