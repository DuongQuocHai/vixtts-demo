import asyncio
import websockets

# Địa chỉ và cổng server sẽ lắng nghe
HOST = '0.0.0.0'
PORT = 9000

# Hàm xử lý mỗi kết nối client (websockets >= 10.0 chỉ truyền websocket)
async def echo_handler(websocket):
    print(f"Client connected: {websocket.remote_address}")
    try:
        async for message in websocket:
            print(f"Received: {message}")
            await websocket.send(f"Echo: {message}")
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected: {websocket.remote_address}")

# Khởi động WebSocket server
if __name__ == "__main__":
    print(f"Starting demo WebSocket server at ws://{HOST}:{PORT}")

    async def main():
        async with websockets.serve(echo_handler, HOST, PORT):
            await asyncio.Future()  # Run forever

    asyncio.run(main())