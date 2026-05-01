import Debug from 'debug';
import { TerminusOptions } from '@godaddy/terminus';

const debug = Debug('express-template:terminus');

function readyCheck() {
  return Promise.resolve();
}

function liveCheck() {
  return Promise.resolve();
}

function onSignal() {
  debug('Server is starting cleanup');
  return Promise.resolve();
}

function beforeShutdown() {
  const timeout = process.env.TERMINATION_GRACE_PERIOD ?? 10000;

  return new Promise((resolve) => {
    setTimeout(resolve, +timeout);
  });
}

function onShutdown() {
  debug('Cleanup finished, server is shutting down');
  return Promise.resolve();
}

export const terminus: TerminusOptions = {
  healthChecks: {
    '/ready': readyCheck,
    '/live': liveCheck,
  },
  beforeShutdown,
  onSignal,
  onShutdown,
};
