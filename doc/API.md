# HTTP API

XMRig's HTTP API is available when XMRig is built with HTTP support (`-DWITH_HTTP=ON`). Enable it in the configuration with `"http": { "enabled": true }`, then choose the `host` and `port`. The API can be protected with a Bearer access token.

Example configuration:

```json
`"api": {
	"id": null,
	"worker-id": null
},
"http": {
	"enabled": true,
	"host": "127.0.0.1",
	"port": 44444,
	"access-token": "SECRET",
	"restricted": true
}
```

#### Global API options
* **id** Miner ID. If it is not set, XMRig creates one automatically.
* **worker-id** Optional worker name. If it is not set, XMRig detects it automatically.

#### HTTP API options
* **enabled** Enable (`true`) or disable (`false`) the HTTP API.
* **host** Host for incoming connections, for example `127.0.0.1` or `0.0.0.0`.
* **port** TCP port for incoming connections. Port `0` is valid and means a random port.
* **access-token** Bearer token used to authenticate API requests. Send it in the `Authorization: Bearer <token>` header.
* **restricted** When `true`, remote configuration changes are disabled. When `false`, configuration update endpoints are allowed.

The equivalent command line options are `--api-id`, `--api-worker-id`, `--http-enabled`, `--http-host`, `--http-port`, `--http-access-token`, and `--http-no-restricted`.

## Authentication and access

Every request is checked for the configured Bearer token when an access token is configured. Requests without the required token receive HTTP `401`; an invalid token receives HTTP `403`.

When the HTTP API is in restricted mode:

* `GET` requests for normal status information are allowed.
* Configuration reads through `/1/config` or `/2/config` return HTTP `403`.
* Configuration writes and JSON-RPC requests are blocked because they are non-`GET` requests.

Non-`GET` requests must use `Content-Type: application/json`.

## Endpoints

### GET /1/summary

Legacy version of the miner summary endpoint. `/2/summary` is preferred for new clients.

### GET /2/summary

Returns a miner summary. The response contains general API information together with miner, hashrate, connection, results, and resource information when those components are available.

The top-level response includes fields such as:

* `id` — API instance ID.
* `worker_id` — configured or detected worker ID.
* `uptime` — miner uptime in seconds.
* `restricted` — whether the current HTTP API request is restricted.
* `resources` — process/system resource information.
* `features` — enabled XMRig features.
* `miner` — miner version, platform information, supported algorithms, donation level, and paused state.
* `hashrate` — total and highest hashrate; version 1 also exposes per-thread values.
* `results` — pool result statistics.
* `connection` — current pool connection information.

Example:

```console
curl http://127.0.0.1:44444/2/summary
```

The historical `/api.json` URL is also accepted as an alias for the summary endpoint.

### GET /1/threads

This endpoint is obsolete. It was replaced by `/2/backends`. New clients should use `/2/backends`.

### GET /2/backends

Returns an array containing the JSON description of each enabled mining backend.

Example:

```console
curl http://127.0.0.1:44444/2/backends
```

The returned objects are backend-specific and may include CPU, OpenCL, or CUDA information.

## Configuration endpoints

The configuration endpoints expose the active miner configuration and, when unrestricted, allow the running configuration to be replaced.

### GET /1/config

Returns the current active miner configuration. `/2/config` is the preferred endpoint for new clients.

```console
curl http://127.0.0.1:44444/1/config
```

### GET /2/config

Returns the current active miner configuration.

```console
curl http://127.0.0.1:44444/2/config
```

This endpoint requires unrestricted access. Use an access token when exposing the API beyond localhost.

### PUT /1/config

Replaces the active miner configuration.

### PUT /2/config

Replaces the active miner configuration. The request body must contain a valid JSON configuration accepted by XMRig.

```console
curl -v --data-binary @config.json -X PUT \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer SECRET" \
  http://127.0.0.1:44444/2/config
```

A successful configuration update returns HTTP `204 No Content`. Invalid JSON or an invalid configuration returns HTTP `400`.

### POST /1/config and POST /2/config

POST is also accepted for configuration updates and follows the same restrictions and response codes as PUT.

## JSON-RPC

### POST /json_rpc

XMRig exposes a small JSON-RPC 2.0 interface for runtime control. The currently supported methods are:

* `pause` — pause mining.
* `resume` — resume mining.

A request uses the JSON-RPC 2.0 format:

```json
{
	"jsonrpc": "2.0",
	"id": 1,
	"method": "pause"
}
```

For a successful request, XMRig returns:

```json
{
	"result": {
		"status": "OK"
	},
	"jsonrpc": "2.0",
	"id": 1
}
```

Example:

```console
curl -X POST http://127.0.0.1:44444/json_rpc \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer SECRET" \
  -d '{"jsonrpc":"2.0","id":1,"method":"pause"}'
```

Unknown methods return JSON-RPC error `-32601` (`Method not found`). Malformed JSON returns `-32700` (`Parse error`), and an invalid request returns `-32600` (`Invalid Request`).
