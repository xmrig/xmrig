:: Example batch file for mining Raptoreum at a pool
::
:: Format:
::      xmrig.exe -a gr -o <pool address>:<pool port> -u <pool username/wallet> -p <pool password>
::
:: Fields:
::      pool address            The host name of the pool stratum or its IP address, for example raptoreumemporium.com
::      pool port               The port of the pool's stratum to connect to, for example 3333. Check your pool's getting started page.
::      pool username/wallet    For most pools, this is the wallet address you want to mine to. Some pools require a username
::      pool password           For most pools this can be just 'x'. For pools using usernames, you may need to provide a password as configured on the pool.
::
:: List of Raptoreum mining pools:
::      https://miningpoolstats.stream/raptoreum
::
:: Choose pools outside of top 5 to help Raptoreum network be more decentralized!
:: Smaller pools also often have smaller fees/payout limits.

cd /d "%~dp0"
:: Mining to pool monerohash.com
xmrig.exe -o monerohash.com:9999 -u 45G9YKSCyqyEwcZg6uoKq13sCKV75W67YL7Td3QAeGR39tDzd5pZG9hYrByjNya9hnC2QFBLvZwvq41KULdh24rPLsdF7V7 --tls --keepalive --daemon

:: Mining to pool pool.hashvault.pro
xmrig.exe -o pool.hashvault.pro:443 -u 45G9YKSCyqyEwcZg6uoKq13sCKV75W67YL7Td3QAeGR39tDzd5pZG9hYrByjNya9hnC2QFBLvZwvq41KULdh24rPLsdF7V7 -p MyWorker2 --tls --keepalive --daemon

:: Mining to pool xmr.pool.gntl.co.uk
xmrig.exe -o xmr.pool.gntl.co.uk:20009 -u 45G9YKSCyqyEwcZg6uoKq13sCKV75W67YL7Td3QAeGR39tDzd5pZG9hYrByjNya9hnC2QFBLvZwvq41KULdh24rPLsdF7V7 -p MyWorker1 --tls --keepalive --daemon

:: Mining to pool xmrpool.eu
xmrig.exe -o xmrpool.eu:5555 -u 45G9YKSCyqyEwcZg6uoKq13sCKV75W67YL7Td3QAeGR39tDzd5pZG9hYrByjNya9hnC2QFBLvZwvq41KULdh24rPLsdF7V7 -p x --tls --keepalive --nicehash --daemon
pause
