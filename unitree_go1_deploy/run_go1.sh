#!/usr/bin/env bash
#
# create_two_tmux_sessions.sh
#
# This script creates (or restarts) two detached tmux sessions,
# each running one of the specified commands.

sudo ip route add 224.0.0.0/4 dev eth0 metric 80

# 1. Session “lcm_position” running the lcm_position binary
tmux has-session -t lcm_position 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Starting lcm_position"
  tmux new-session -d -s lcm_position \
    zsh -c "cd \"$HOME/unitree_go1_deploy/unitree_legged_sdk/build\" && ./lcm_position"
fi

# 2. Session “go1_deploy” deactivating conda, then running deploy.py
tmux has-session -t go1_deploy 2>/dev/null
if [ $? -ne 0 ]; then
  echo "Starting go1_deploy"
  tmux new-session -d -s go1_deploy \
    zsh -c "cd $HOME/unitree_go1_deploy/go1_deploy/go1_gym_deploy/scripts && /usr/bin/python3 deploy.py"
fi

echo "Two tmux sessions are now running (or have been restarted):"
tmux ls
echo "Use 'tmux ls' to list them and 'tmux attach -t <session_name>' to attach. Use 'tmux kill-server' to kill all sessions."