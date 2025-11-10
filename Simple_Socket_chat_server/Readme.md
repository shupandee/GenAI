# Simple Socket Chat Server

A TCP-based multi-client chat server implementation using Python's standard library. This server supports real-time messaging, private messages, user management, and automatic idle timeout.

## Features

- ✅ Multi-client support (handles 10+ concurrent connections)
- ✅ User authentication with unique usernames
- ✅ Real-time message broadcasting
- ✅ Private messaging (DM) between users
- ✅ Active user listing
- ✅ Heartbeat/ping functionality
- ✅ Automatic idle timeout (60 seconds)
- ✅ Clean disconnect notifications

## Requirements

- Python 3.6 or higher
- No external dependencies (uses only standard library)

## Installation

1. Clone or download the project files:
```bash
git clone <repository-url>
cd socket-chat-server
```

2. Ensure you have the following files:
   - `chat_server.py` - The server implementation
   - `chat_client.py` - The client implementation

## Running the Server

### Default Configuration (Port 4000)

```bash
python chat_server.py
```

### Custom Port

```bash
# Using command-line argument
python chat_server.py 5000

# Using environment variable
export CHAT_PORT=5000
python chat_server.py
```

The server will start and display:
```
[SERVER] Chat server started on 0.0.0.0:4000
[SERVER] Waiting for connections...
```

## Connecting to the Server

### Option 1: Using the Python Client

```bash
python chat_client.py [host] [port]

# Examples:
python chat_client.py                    # Connect to localhost:4000
python chat_client.py localhost 5000     # Connect to localhost:5000
python chat_client.py 192.168.1.100 4000 # Connect to remote server
```

### Option 2: Using netcat (nc)

```bash
nc localhost 4000
```

### Option 3: Using telnet

```bash
telnet localhost 4000
```

## Protocol Commands

### Login (Required First Step)
```
LOGIN <username>
```
**Response:**
- `OK` - Login successful
- `ERR username-taken` - Username already in use
- `ERR empty-username` - No username provided

### Send Message to All Users
```
MSG <text>
```
**Everyone else receives:**
```
MSG <username> <text>
```

### Send Private Message
```
DM <username> <text>
```
**Target user receives:**
```
DM <sender> <text>
```
**Sender receives:**
```
DM-SENT <username>
```

### List Active Users
```
WHO
```
**Response:**
```
USER <username1>
USER <username2>
...
```

### Heartbeat/Ping
```
PING
```
**Response:**
```
PONG
```

### Quit (Python Client Only)
```
quit
```

## Example Usage

### Terminal 1 (User: Alice)
```bash
$ python chat_client.py
Connected to chat server at localhost:4000
==================================================

Commands:
  LOGIN <username>  - Log in with a username
  MSG <text>        - Send a message to everyone
  DM <user> <text>  - Send private message
  WHO               - List active users
  PING              - Test connection
  quit              - Exit chat
==================================================

LOGIN Alice
OK
MSG Hello everyone!
MSG Bob hello Naman!
DM Bob Hey, how are you?
DM-SENT Bob
INFO Bob disconnected
```

### Terminal 2 (User: Bob)
```bash
$ python chat_client.py
Connected to chat server at localhost:4000
==================================================

Commands:
  LOGIN <username>  - Log in with a username
  MSG <text>        - Send a message to everyone
  DM <user> <text>  - Send private message
  WHO               - List active users
  PING              - Test connection
  quit              - Exit chat
==================================================

LOGIN Bob
OK
MSG Alice Hello everyone!
MSG Hi Alice!
DM Alice Hey, how are you?
WHO
USER Alice
USER Bob
quit
Goodbye!
```

### Terminal 3 (User: Charlie - Using netcat)
```bash
$ nc localhost 4000
LOGIN Charlie
OK
MSG Hey guys!
MSG Alice Hey guys!
MSG Bob Hi Alice!
INFO Bob disconnected
^C
```

## Error Handling

The server handles various error conditions:

- **Username already taken:** `ERR username-taken`
- **Empty username:** `ERR empty-username`
- **User not found (DM):** `ERR user-not-found`
- **Invalid command:** `ERR unknown-command`
- **Invalid DM format:** `ERR invalid-dm-format`
- **Empty message:** `ERR empty-message`
- **Idle timeout:** User disconnected after 60 seconds of inactivity

## Architecture

### Server (`chat_server.py`)
- Multi-threaded architecture using Python's `threading` module
- Thread-safe client management with locks
- Automatic idle timeout checking (every 10 seconds)
- Clean disconnect handling

### Client (`chat_client.py`)
- Separate threads for sending and receiving messages
- User-friendly command interface
- Graceful shutdown on quit or Ctrl+C

## Demo

### Screen Recording

🎥 **Demo Video:** [Watch on Google Drive](https://drive.google.com/file/d/1R4Qll3tvBidwsxVv82WcCzINlugiZwfr/view?usp=sharing)

The video demonstrates:
- Starting the server
- Multiple clients connecting simultaneously
- Public messaging between users
- Private messaging (DM)
- User list command (WHO)
- User disconnection notifications
- Idle timeout functionality

### Screenshot

![Chat Server Demo](server.png)

*Three terminals showing the server and two clients (Deepanshu and Ansh) chatting in real-time, demonstrating public messages, private messages, and the WHO command.*

## Testing

To test the server with multiple clients:

1. Start the server in one terminal
2. Open 3+ additional terminals
3. Connect each terminal as a different user
4. Test various commands and scenarios

### Test Scenarios
- ✅ Multiple users login simultaneously
- ✅ Broadcast messages to all users
- ✅ Private messages between specific users
- ✅ List active users
- ✅ User disconnection and notifications
- ✅ Idle timeout (wait 60+ seconds without activity)
- ✅ Username collision (try same username twice)

## Deployment

### Local Network Deployment
```bash
# Find your local IP address
# Linux/Mac: ifconfig or ip addr
# Windows: ipconfig

# Run server bound to all interfaces
python chat_server.py

# Clients connect using your IP
python chat_client.py 192.168.1.XXX 4000
```

### Cloud Deployment (Optional)
If deployed to a cloud server (AWS EC2, DigitalOcean, etc.):
- Ensure port 4000 is open in firewall/security groups
- Use the server's public IP address
- Consider using a process manager like `systemd` or `supervisor`

## Limitations

- Text-based messages only (no file transfer)
- No message history (messages are not persisted)
- No encryption (plaintext communication)
- Simple authentication (username-only, no passwords)

## Future Enhancements

- [ ] Persistent message history with database
- [ ] User authentication with passwords
- [ ] Encrypted communication (TLS/SSL)
- [ ] Chat rooms/channels
- [ ] File transfer support
- [ ] Web-based client interface

## License

This project is created for educational purposes.

## Author

**Deepanshu Gautam** - Backend Assignment

## Contact

For questions or issues, please contact [shupandee@gmail.com]

---

**Note:** This implementation uses only Python's standard library as required by the assignment specifications.
