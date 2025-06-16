# HTTP API

If you want use HTTP API you need enable it (`"enabled": true,`) then choice `port` and optionaly `host`. API not available if miner built without HTTP support (`-DWITH_HTTP=OFF`).

Offical HTTP client for API: http://workers.xmrig.info/

Example configuration, used in Curl examples below:

```json
"api": {
	"id": null,
	"worker-id": null,
},
"http": {
	"enabled": true,
	"host": "127.0.0.1",
	"port": 44444,
	"access-token": "SECRET",
	"restricted": false
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

### APIVersion 2

#### GET /2/summary

Get miner summary information. [Example](api/2/summary.json).

#### GET /2/backends

Get detailed information about miner backends. [Example](api/2/backends.json).

### APIVersion 1 (deprecated)

#### GET /1/summary

Get miner summary information. Currently identical to `GET /2/summary`

#### GET /1/threads

**REMOVED** Get detailed information about miner threads. [Example](api/1/threads.json).

Functionally replaced by `GET /2/backends` which contains a `threads` item per backend.

### APIVersion 0 (deprecated)

#### GET /api.json

Get miner summary information. Currently identical to `GET /2/summary`


## Restricted endpoints

All API endpoints below allow access to sensitive information and remote configure miner. You should set `access-token` and allow unrestricted access (`"restricted": false`).

### JSON-RPC Interface

#### POST /json_rpc

Control miner with JSON-RPC. Methods: `pause`, `resume`, `stop`, `start`

Curl example:

```
curl -v --data "{\"method\":\"pause\",\"id\":1}" -H "Content-Type: application/json" -H "Authorization: Bearer SECRET" http://127.0.0.1:44444/json_rpc
```

### APIVersion 2

#### GET /2/config

Get current miner configuration. [Example](api/2/config.json).

#### PUT /2/config

Update current miner configuration. Common use case, get current configuration, make changes, and upload it to miner.

Curl example:

```
...GET current config...
curl -v -H "Content-Type: application/json" -H "Authorization: Bearer SECRET" http://127.0.0.1:44444/2/config > config.json
...make changes...
vim config.json
...PUT changed config...
curl -v --data-binary @config.json -X PUT -H "Content-Type: application/json" -H "Authorization: Bearer SECRET" http://127.0.0.1:44444/2/config
```

### APIVersion 1 (deprecated)

#### GET /1/config

Get current miner configuration. Currently identical to `GET /2/config`

#### PUT /1/config

Update current miner configuration. Currently identical to `PUT /2/config`
