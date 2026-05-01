FROM node:20-alpine as parent

FROM parent as build

ARG NODE_AUTH_TOKEN

# hadolint ignore=DL3018
RUN apk add --no-cache python3 make g++

WORKDIR /usr/src/app

COPY package.json package-lock.json tsconfig.json ./
RUN npm ci && rm -f .npmrc

COPY src/ ./src/
COPY bin/ ./bin/

RUN npm run build

###

FROM parent as base

USER node
WORKDIR /home/node/app

COPY --from=build --chown=node:node /usr/src/app/ ./

ENV TZ=Europe/Berlin

EXPOSE 4000

###

FROM base as test
ENV NODE_ENV=test

ARG SKIP_TESTS=false

COPY --chown=node:node jest.config.ts ./
COPY --chown=node:node test/ ./test/

RUN if [ "${SKIP_TESTS}" != "true" ]; then npm run test; fi

CMD [ "npm", "run", "test" ]

###

FROM base as development
ENV NODE_ENV=development

EXPOSE 9229

CMD [ "npm", "run", "start:dev" ]

###

FROM parent
ENV NODE_ENV=production

ARG NODE_AUTH_TOKEN

WORKDIR /home/node/app

COPY --from=build --chown=node:node /usr/src/app/package.json ./
COPY --from=build --chown=node:node /usr/src/app/package-lock.json ./
COPY --from=build --chown=node:node /usr/src/app/dist/ ./

RUN npm pkg set scripts.postinstall="echo no-postinstall" \
 && npm ci --omit=dev \
 && rm -f .npmrc package-lock.json

USER node

CMD [ "node", "./bin/www.js" ]
