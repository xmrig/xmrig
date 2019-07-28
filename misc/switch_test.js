#!/usr/bin/env node

// Miner Tester: testing miner algo switch stability

// sudo apt-get update -y
// sudo apt-get install -y git build-essential cmake libuv1-dev libmicrohttpd-dev libssl-dev libhwloc-dev nodejs
// git clone https://github.com/MoneroOcean/xmrig.git
// cd xmrig
// cmake .
// make -j`nproc`
// cp src/config.json .
// sed -i 's/"url": *"[^"]*",/"url": "localhost:3333",/' config.json
// sed -i 's/"user": *"[^"]*",/"user": "44qJYxdbuqSKarYnDSXB6KLbsH4yR65vpJe3ELLDii9i4ZgKpgQXZYR4AMJxBJbfbKZGWUxZU42QyZSsP4AyZZMbJBCrWr1",/' config.json

"use strict";

// *****************************************************************************
// *** DEPENDECIES                                                           ***
// *****************************************************************************

const net = require('net');

// *****************************************************************************
// *** CONSTS                                                                ***
// *****************************************************************************

const algos = [ "rx/wow", "rx/loki", "cn/r" ];

// *****************************************************************************
// *** WORKING STATE                                                         ***
// *****************************************************************************

let curr_miner_socket = null;

// *****************************************************************************
// *** FUNCTIONS                                                             ***
// *****************************************************************************

// *** Console/log output

function log(msg) {
  console.log(">>> " + msg);
}

function err(msg) {
  console.error("!!! " + msg);
}

// *** Miner socket processing

const test_blob_str = "7f7ffeeaa0db054f15eca39c843cb82c15e5c5a7743e06536cb541d4e96e90ffd31120b7703aa90000000076a6f6e34a9977c982629d8fe6c8b45024cafca109eef92198784891e0df41bc03";

function makehex(length) {
   var result           = '';
   var characters       = 'abcdef0123456789';
   var charactersLength = characters.length;
   for ( var i = 0; i < length; i++ ) {
      result += characters.charAt(Math.floor(Math.random() * charactersLength));
   }
   return result;
}

let miner_server = net.createServer(function (miner_socket) {
  if (curr_miner_socket) {
    err("Miner server on localhost:3333 port is already connected (please make sure you do not have other miner running)");
    return;
  }
  log("Miner server on localhost:3333 port connected from " + miner_socket.remoteAddress);

  let miner_data_buff = "";

  miner_socket.on('data', function (msg) {
    miner_data_buff += msg;
    if (miner_data_buff.indexOf('\n') === -1) return;
    let messages = miner_data_buff.split('\n');
    let incomplete_line = miner_data_buff.slice(-1) === '\n' ? '' : messages.pop();
    for (let i = 0; i < messages.length; i++) {
      let message = messages[i];
      if (message.trim() === '') continue;
      let json;
      try {
        json = JSON.parse(message);
      } catch (e) {
        err("Can't parse message from the miner: " + message);
        continue;
      }
      const is_keepalived = "method" in json && json.method === "keepalived";
      if ("method" in json && json.method === "login") {
        miner_socket.write(
          '{"id":1,"jsonrpc":"2.0","error":null,"result":{"id":"benchmark","job":{"blob":"' + test_blob_str +
          '","algo":"cn/wow","job_id":"benchmark1","target":"10000000","height":10,"seed_hash":"' + makehex(64) + '","id":"benchmark"},"status":"OK"}}\n'
        );
        curr_miner_socket = miner_socket;
      }
    }
    miner_data_buff = incomplete_line;
  });
  miner_socket.on('end', function() {
    log("Miner socket was closed");
    curr_miner_socket = null;
  });
  miner_socket.on('error', function() {
    err("Miner socket error");
    miner_socket.destroy();
    curr_miner_socket = null;
  });
});

let job_num = 1;
function change_algo() {
  if (curr_miner_socket) {
    const algo = algos[Math.floor(Math.random() * algos.length)];
    log("Switching to " + algo);
    curr_miner_socket.write(
      '{"jsonrpc":"2.0","method":"job","params":{"blob":"' + test_blob_str + '","algo":"' + algo +
      '","job_id":"benchmark' + ++job_num + '","height":10,"seed_hash":"' + makehex(64) + '","target":"10000000","id":"benchmark"}}\n'
    );
  }
  const sleep = Math.floor(15 + Math.random() * 10);
  log("Waiting " + sleep + "s");
  setTimeout(change_algo, sleep * 1000);
}

miner_server.listen(3333, "localhost", function() {
  log("Local miner server on localhost:3333 port started");
  change_algo();
});