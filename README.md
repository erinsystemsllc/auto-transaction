## Get started

Run `docker-compose up`

If you use packages from our private npm registry,
you need to export an authentication token:

`export NODE_AUTH_TOKEN=<TOKEN>`

or directly

`NODE_AUTH_TOKEN=<TOKEN> docker-compose up` (or build)

#### Starting the dev server without docker
Create a `.env` file with:

```
NODE_ENV=development
```

and run

```
npm run start:dev
```
