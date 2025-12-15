#!/usr/bin/env python3
"""
ZTE System Supervisor
Manages api_server_trading.py and auto_trader_tws.py as supervised processes.
Auto-restarts on crashes to ensure system stability.
"""

import subprocess
import time
import sys
import signal
from datetime import datetime
from pathlib import Path

# Process configuration
PROCESSES = {
    'api_server': {
        'name': 'API Server',
        'command': [sys.executable, 'api_server_trading.py'],
        'startup_delay': 5,  # Seconds to wait after starting
        'restart_delay': 3,  # Seconds to wait before restart
        'process': None,
        'restarts': 0
    },
    'trader_bot': {
        'name': 'Trading Bot',
        'command': [sys.executable, 'auto_trader_tws.py'],
        'startup_delay': 0,
        'restart_delay': 3,
        'process': None,
        'restarts': 0
    }
}

# Global shutdown flag
shutdown_requested = False

def log(message: str, level: str = "INFO"):
    """Log timestamped message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

def start_process(proc_key: str) -> bool:
    """Start a supervised process."""
    proc_info = PROCESSES[proc_key]

    try:
        log(f"Starting {proc_info['name']}...", level="INFO")

        # Start process
        proc = subprocess.Popen(
            proc_info['command'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            cwd=Path(__file__).parent
        )

        proc_info['process'] = proc
        log(f"{proc_info['name']} started (PID: {proc.pid})", level="INFO")

        # Wait for startup delay
        if proc_info['startup_delay'] > 0:
            log(f"Waiting {proc_info['startup_delay']}s for {proc_info['name']} to initialize...", level="DEBUG")
            time.sleep(proc_info['startup_delay'])

        return True

    except Exception as e:
        log(f"Failed to start {proc_info['name']}: {e}", level="ERROR")
        return False

def stop_process(proc_key: str, timeout: int = 10):
    """Stop a supervised process gracefully."""
    proc_info = PROCESSES[proc_key]
    proc = proc_info['process']

    if proc is None:
        return

    try:
        log(f"Stopping {proc_info['name']} (PID: {proc.pid})...", level="INFO")

        # Send SIGTERM for graceful shutdown
        proc.terminate()

        # Wait for process to exit
        try:
            proc.wait(timeout=timeout)
            log(f"{proc_info['name']} stopped gracefully", level="INFO")
        except subprocess.TimeoutExpired:
            log(f"{proc_info['name']} did not stop gracefully - forcing kill", level="WARNING")
            proc.kill()
            proc.wait()
            log(f"{proc_info['name']} force killed", level="WARNING")

    except Exception as e:
        log(f"Error stopping {proc_info['name']}: {e}", level="ERROR")

    finally:
        proc_info['process'] = None

def check_process(proc_key: str) -> bool:
    """Check if process is running. Returns True if alive, False if dead."""
    proc_info = PROCESSES[proc_key]
    proc = proc_info['process']

    if proc is None:
        return False

    # Check if process has exited
    exit_code = proc.poll()

    if exit_code is not None:
        # Process died
        proc_info['restarts'] += 1
        log(f"{proc_info['name']} died (exit code: {exit_code}) - Restart #{proc_info['restarts']}", level="ERROR")
        proc_info['process'] = None
        return False

    return True

def handle_shutdown(signum, frame):
    """Handle shutdown signals."""
    global shutdown_requested
    log("Shutdown signal received", level="INFO")
    shutdown_requested = True

def main():
    """Main supervisor loop."""
    global shutdown_requested

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    log("=== ZTE System Supervisor Starting ===", level="INFO")
    log("Press Ctrl+C to shutdown gracefully", level="INFO")

    # Start API Server first
    if not start_process('api_server'):
        log("Failed to start API Server - aborting", level="ERROR")
        return 1

    # Start Trading Bot
    if not start_process('trader_bot'):
        log("Failed to start Trading Bot - stopping API Server", level="ERROR")
        stop_process('api_server')
        return 1

    log("=== All systems running ===", level="INFO")

    # Monitoring loop
    check_interval = 10  # Check every 10 seconds
    last_status_log = time.time()

    try:
        while not shutdown_requested:
            time.sleep(check_interval)

            # Check API Server
            if not check_process('api_server'):
                if shutdown_requested:
                    break

                log("Restarting API Server...", level="WARNING")
                time.sleep(PROCESSES['api_server']['restart_delay'])

                if not start_process('api_server'):
                    log("Failed to restart API Server - shutting down system", level="ERROR")
                    shutdown_requested = True
                    break

            # Check Trading Bot
            if not check_process('trader_bot'):
                if shutdown_requested:
                    break

                log("Restarting Trading Bot...", level="WARNING")
                time.sleep(PROCESSES['trader_bot']['restart_delay'])

                if not start_process('trader_bot'):
                    log("Failed to restart Trading Bot - shutting down system", level="ERROR")
                    shutdown_requested = True
                    break

            # Status update every 5 minutes
            if time.time() - last_status_log > 300:
                api_restarts = PROCESSES['api_server']['restarts']
                bot_restarts = PROCESSES['trader_bot']['restarts']
                log(f"System Status: API Server (restarts: {api_restarts}), Trading Bot (restarts: {bot_restarts})", level="INFO")
                last_status_log = time.time()

    except KeyboardInterrupt:
        log("Keyboard interrupt received", level="INFO")
        shutdown_requested = True

    # Graceful shutdown
    log("=== Initiating graceful shutdown ===", level="INFO")

    # Stop Trading Bot first (to avoid new trades)
    stop_process('trader_bot', timeout=15)

    # Stop API Server
    stop_process('api_server', timeout=10)

    # Summary
    api_restarts = PROCESSES['api_server']['restarts']
    bot_restarts = PROCESSES['trader_bot']['restarts']
    log(f"=== Shutdown complete ===", level="INFO")
    log(f"Session Summary - API Server restarts: {api_restarts}, Trading Bot restarts: {bot_restarts}", level="INFO")

    return 0

if __name__ == "__main__":
    sys.exit(main())
