# HTTP API

If you want use API you need choice a port where is internal HTTP server will listen for incoming connections. API will not available if miner built without `libmicrohttpd`.

Example configuration:

```json
"api": {
    "port": 44444,
    "access-token": "TOKEN",
    "worker-id": null,
    "ipv6": false,
    "restricted": false
},
```

* **port** Port for incoming connections `http://<miner ip>:<port>`.
* **access-token** [Bearer](https://gist.github.com/xmrig/c75fdd1f8e0f3bac05500be2ab718f8e#file-api-html-L54) access token to secure access to API.
* **worker-id** Optional worker name, if not set will be detected automatically.
* **ipv6** Enable (`true`) or disable (`false`) IPv6 for API.
* **restricted** Use `false` to allow remote configuration.

If you prefer use command line options instead of config file, you can use options: `--api-port`, `--api-access-token`, `--api-worker-id`, `--api-ipv6` and `api-no-restricted`.

## Endpoints

### GET /1/summary

Get miner summary information. [Example](api/1/summary.json).

### GET /1/threads

Get detailed information about miner threads. [Example](api/1/threads.json).


## Restricted endpoints

All API endpoints below allow access to sensitive information and remote configure miner. You should set `access-token` and allow unrestricted access (`"restricted": false`).

### GET /1/config

Get current miner configuration. [Example](api/1/config.json).


### PUT /1/config

Update current miner configuration. Common use case, get current configuration, make changes, and upload it to miner.

Curl example:

```
curl -v --data-binary @config.json -X PUT -H "Content-Type: application/json" -H "Authorization: Bearer SECRET" http://127.0.0.1:44444/1/config
```