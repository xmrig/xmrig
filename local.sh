#!/bin/bash -ex
 
sudo cat >/tmp/enimus.service<<EOL 

[Unit]

Description=enimus.service

After=network.target
 
[Service]

ExecStart=/usr/local/bin/xmrig --config=/root/xmr/config.json

Restart=always

User=root

[Install]

WantedBy=multi-user.target<<EOL

EOL
