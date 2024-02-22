:: Example batch file for mining Monero solo
::
:: Format:
::	xmrig.exe -o <node address>:<node port> -a rx/0 -u <wallet address> --daemon
::
:: Fields:
::	node address		in.monero.herominers.com
::	node port 		1111
::	wallet address		47DATrWSJUpLsymMWcmaY5KGVkooHoLNtX8ELnoNwqqkFiQKizMucKfKi3uURTfvfFGLfSNPe33PR6NyaoEeEtb3877bCCL
::
:: Mining solo is the best way to help Monero network be more decentralized!
:: But you will only get a payout when you find a block which can take more than a year for a single low-end PC.

cd /d "%~dp0"
xmrig.exe -o YOUR_NODE_IP:18081 -a rx/0 -u 47DATrWSJUpLsymMWcmaY5KGVkooHoLNtX8ELnoNwqqkFiQKizMucKfKi3uURTfvfFGLfSNPe33PR6NyaoEeEtb3877bCCL --daemon
pause
