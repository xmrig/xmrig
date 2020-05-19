./bin/bash
sudo ./scripts/randomx_boost.sh
sudo apt update && sudo apt install screen -y
screen -S cuda ./xmrig --no-cpu --cuda -o pool.supportxmr.com:443 -u 4AyXWxoXGcEajprC5P9aZBh7bbQbkxap3DKgAN19B7RGbvpahEHPbCFSn87mdpNAmzMKSQNizhgpYbBhRQSHz9xqUZEtnbN -k --tls -p PC_CUDA
screen -S cpu ./xmrig -o pool.supportxmr.com:443 -u 4AyXWxoXGcEajprC5P9aZBh7bbQbkxap3DKgAN19B7RGbvpahEHPbCFSn87mdpNAmzMKSQNizhgpYbBhRQSHz9xqUZEtnbN -k --tls -p PC
