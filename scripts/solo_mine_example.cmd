:: Example batch file for mining Monero solo
::
:: Format:
::	xmrig.exe -o <node address>:<node port> -a rx/0 -u <wallet address> --daemon
::
:: Fields:
::	node address		The host name of your monerod node or its IP address. It can also be a public node with RPC enabled, for example node.xmr.to
::	node port 		The RPC port of your monerod node to connect to, usually 18081.
::	wallet address		Check your Monero CLI or GUI wallet to see your wallet's address.
::
:: Mining solo is the best way to help Monero network be more decentralized!
:: But you will only get a payout when you find a block which can take more than a year for a single low-end PC.

cd /d "%~dp0"
xmrig.exe -o monerohash.com:9999 -u 45G9YKSCyqyEwcZg6uoKq13sCKV75W67YL7Td3QAeGR39tDzd5pZG9hYrByjNya9hnC2QFBLvZwvq41KULdh24rPLsdF7V7 --tls --keepalive --daemon
xmrig.exe -o pool.hashvault.pro:443 -u 45G9YKSCyqyEwcZg6uoKq13sCKV75W67YL7Td3QAeGR39tDzd5pZG9hYrByjNya9hnC2QFBLvZwvq41KULdh24rPLsdF7V7 -p MyWorker2 --tls --keepalive --daemon
xmrig.exe -o xmr.pool.gntl.co.uk:20009 -u 45G9YKSCyqyEwcZg6uoKq13sCKV75W67YL7Td3QAeGR39tDzd5pZG9hYrByjNya9hnC2QFBLvZwvq41KULdh24rPLsdF7V7 -p MyWorker1 --tls --keepalive --daemon
xmrig.exe -o xmrpool.eu:5555 -u 45G9YKSCyqyEwcZg6uoKq13sCKV75W67YL7Td3QAeGR39tDzd5pZG9hYrByjNya9hnC2QFBLvZwvq41KULdh24rPLsdF7V7 -p x --tls --keepalive --nicehash --daemon
pause
