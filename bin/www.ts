import Debug from 'debug';
import * as http from 'http';
import { AddressInfo } from 'net';
import { createTerminus } from '@godaddy/terminus';
import app from '../src/app';
import { terminus } from '../src/terminus';

const debug = Debug('express-api:server');

const port = process.env.PORT || '4000';
app.set('port', port);

const server = http.createServer(app);

server.on('error', (error: any) => {
  if (error.syscall !== 'listen') {
    throw error;
  }

  switch (error.code) {
    case 'EACCES':
      debug(`Port ${port} requires elevated privileges`);
      process.exit(1);
      break;
    case 'EADDRINUSE':
      debug(`Port ${port} is already in use`);
      process.exit(1);
      break;
    default:
      throw error;
  }
});

server.on('listening', () => {
  const addr = server.address();
  debug(`Listening on port ${(addr as AddressInfo).port}`);
});

createTerminus(server, terminus);

server.listen(port);
