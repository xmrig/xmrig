# 1.6.4
- Fix connection issues #130
- Remote logging (Miner log on the Dashboard)
- Add resetClientStatusList button to Dashboard #129
- Fix new version notification #132
- Add Masari (MSR) v7 support
- Add Haven Protocol (XHV) v3 support
# 1.6.3
- Added shift+click function for multi row selection to Dashboard
- Added -DBUILD_STATIC=ON/OFF option to CMake configuration to create fully static builds
- Added current algo and list of supported_varaints to login message for future usage on proxy
- Added support for latest Stellite (XTL) and Alloy (XAO) variants
- Simplification of configuration, "force-pow-variant" and "cryptonight-lite-ipbc" parameters are now deprecated see [Coin Configuration](https://github.com/Bendr0id/xmrigCC/wiki/Coin-configurations) for guidance
- Fixed leaks in transport shutdown
# 1.6.2
- Implementation of CN-Lite-IPBC algo
- Fixed Windows 32bit build / crashes
- Fixed XMRigCCServer crash when auth header is manipulated 
# 1.6.1
- beta
# 1.6.0
- Complete rewrite of the stratum TCP/TLS network communication using boost::asio to fix connection issues and crashs 
- Force of PoW via "variant" parameter in xmrg-proxy 2.5.2+, it now overrules local settings
- Implementation of CN-Heavy algo used by Sumokoin / Haven / ... 
- XMRigDaemon now keeps the miner running event when the miner crashs
# 1.5.5
- Fixed Bad/Invalid shares and high share transmit latency
- Fixed hugepages for some older linux versions
- Fixed compatibility to xmrig-proxy 2.5.x+
- Added restart of crashed miners to xmrigDaemon  
- Added force algo variant by xmrig-proxy 2.5.x+
- Added auto force of nicehash param by xmrig-proxy 2.5.x+
- Partial rebase of XMRig 2.5.2
# 1.5.2
- Fixed OSX Build
- Fixed force PoW algo version 
- Added AEON test vectors for new PoW Algo
- Changed DonateStrategy to avoid peaks on donate pool when restarting multiple miners 
# 1.5.1
- Applied changes for upcoming Monero v7 PoW changes starting 03/28/18 (No changes in config needed) 
- Applied changes for upcoming AEON PoW changes starting 04/07/18  (No changes in config needed)
- Added option to force PoW version 
- Added new design / icons
# 1.5.0
- Full SSL/TLS support for the whole communication:
    - XMRigCCServer Dashboard <-> Browser
    - XMRigCCServer <-> XMRigMiner
    - XMRigMiner <-> Pool
- Easy rename of miner/daemon in CMakeList.txt by modifying `MINER_EXECUTABLE_NAME` and `DAEMON_EXECUTABLE_NAME` before compiling
- Dockerfile and official DockerHub image
- Added Miner uptime to Dashboard
- Rebased from XMRig 2.4.5 RC
# 1.4.0
- Fixed CPU affinity on Windows for NUMA and CPUs with lot of cores
- Implemented per thread configurable Multihash mode (double, triple, quadruple, quintuple)
- Rebased from XMRig 2.4.4
# v1.3.2
- Added start xmrigDaemonr without config file and only CCServer/auth token as params needed #14
- Dashboard now uses servertime for calculation to avoid clock drifts and false offline detection
- Finally fixed freebsd build
# v1.3.1
- Removed not working background mode for xmrigMiner/xmrigDaemon on *nix systems -> use screen/tmux or systemd service instead
- Added cpu socket to client Id tooltip on dashboard
- Fixed notification when sending command is successful or error
- Fixed #16 FreeBSD build
- Fixed miner to keep sending status to server when its not temp unavailable
- Fixed #10 CCServer spontaneously freezes and holds CPU 100%
- Merged latest xmrig master
# v1.3.0
- Fixed Soft-aes modes (av=3, av=4) Bug: #11
- Added static build for linux with old libc (CentOs 5/6, debian, ...)
- Added notification to Dashboard when miner went offline with toggleswitch
- Added multi config editor to Dashboard to modify config of multiple miners at once
- Fixed MSV_VER for latest Visual Studio builds
# v1.2.2
- Added select/deselect all to dashboard
- Fixed memory leaks in XmrigCCServer
# v1.2.1
- Refactored Dashboard to send one command to multiple miners and "beautified" dashboard
- Miner now publishs own config to XMRigCCServer on startup
- Added command to trigger miner to upload config to XMRigCCServer
- Added threads to miner info tooltip on client id
# v1.2.0
- Added configurability for thread based doublehash mode which helps you to use more of your l3 cache
- Memory optimizations / speed improvements
- Updated to latest XMRig (2.4.3) with ARM support
# v1.1.1
- Fixed table sorting for client id column on Dashboard
- Fixed Windows compilation with msys2 (gcc)
- Added ability to do static build of xmrigDaemon and xmrigMiner
- Added client version to Dashboard client id tooltip
- Added update checker to Dashboard with notification bar 
# v1.1.0
- Added option to hide offline miners from Dashboard
- Added online status indicator to Dashboard client id column (green:red)
- JSON-Protocol changes to send miner info to XMRigCC server
- Added Tooltip to Dashboard column id containing new miner info (CPU, CPU flags, Cores, Threads, Memory, External IP, ...)
- Moved CCClient to own thread and changed ControlCommand processing to async to improve performance
# v1.0.9
- Integrated cpp-httplib as libcurl replacement 
- Removed libcurl dependicies
- Fixed round of avgTime in Dashboard
- Removed subrepo dependencies for easier building
# v1.0.8
- Extracted common CC files to subrepo (xmrigCC-common)
- Added sum row to Dashboard
- Added dialogs (success/error) to all dashboard actions
# v1.0.7
- CCClient status update interval is now configurable
- Updated to latest head of xmrig (Optimized soft aes)
# v1.0.6
- Fixed launch in folder containing spaces (Win)
# v1.0.5
- Merged latest changes of XMRig
- Added current algo to dashboard
- Added editor for client configs to dashboard
- some cosmetics on dashboard
# v1.0.4
- Updated XMRig to 2.4.2
- Fixed "--background" not working for xmrigCCServer on Windows
# v1.0.3
- Integrated build for Windows x64
# v1.0.2
- Reenabled restart in daemon for win/linux
# v1.0.1
- Fixed windows build
# v1.0.0
- Initial public release based on xmrig version 2.4.1
