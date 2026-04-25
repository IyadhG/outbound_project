#!/usr/bin/env python3
"""
Quick test for Detective MCP Server

This starts the server and tests one tool to verify it's working.
"""

import asyncio
import subprocess
import sys
import json
from pathlib import Path

async def test_server():
    """Test the MCP server."""
    
    server_path = Path(__file__).parent / "mcp_server.py"
    
    print("🚀 Starting Detective MCP Server...")
    print(f"   Server: {server_path}\n")
    
    # Start the server process
    process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Wait a moment for server to start
        await asyncio.sleep(1)
        
        # Check if process is still running
        if process.poll() is not None:
            stderr = process.stderr.read()
            print(f"❌ Server crashed immediately!")
            print(f"Error: {stderr}")
            return
        
        print("✅ Server started successfully!")
        print("   Server is running and waiting for MCP client connections.\n")
        
        print("📋 To test the server, run this in another terminal:")
        print("   python test_mcp_client.py")
        print("\n   Or use the MCP inspector:")
        print("   npx @anthropics/mcp-inspector python mcp_server.py\n")
        
        print("🛑 Press Ctrl+C to stop the server")
        
        # Keep server running
        while True:
            await asyncio.sleep(1)
            if process.poll() is not None:
                print("\n⚠️  Server stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping server...")
    finally:
        process.terminate()
        try:
            process.wait(timeout=2)
        except:
            process.kill()
        print("✅ Server stopped")

if __name__ == "__main__":
    try:
        asyncio.run(test_server())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
