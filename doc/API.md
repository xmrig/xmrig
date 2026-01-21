# HTTP API

If you want use HTTP API you need enable it (`"enabled": true,`) then choice `port` and optionaly `host`. API not available if miner built without HTTP support (`-DWITH_HTTP=OFF`).

Offical HTTP client for API: http://workers.xmrig.info/

Example configuration:

```json
"api": {
	"id": null,
	"worker-id": null,
},
"http": {
	"enabled": false,
	"host": "127.0.0.1",
	"port": 0,
	"access-token": null,
	"restricted": true
}
```

#### Global API options
* **id** Miner ID, if not set created automatically.
* **worker-id** Optional worker name, if not set will be detected automatically.

#### HTTP API options,
* **enabled** Enable (`true`) or disable (`false`) HTTP API.
* **host** Host for incoming connections `http://<host>:<port>`, to allow connections from all interfaces use `0.0.0.0` (IPv4) or `::` (IPv4+IPv6).
* **port** Port for incoming connections `http://<host>:<port>`, zero port is valid option and means random port.
* **access-token** [Bearer](https://gist.github.com/xmrig/c75fdd1f8e0f3bac05500be2ab718f8e#file-api-html-L54) access token to secure access to API. Miner support this token only via `Authorization` header.
* **restricted** Use `false` to allow remote configuration.

If you prefer use command line options instead of config file, you can use options: `--api-id`, `--api-worker-id`, `--http-enabled`, `--http-host`, `--http-access-token`, `--http-port`, `--http-no-restricted`.

Versions before 2.15 was use another options for API https://github.com/xmrig/xmrig/issues/1007

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
