FROM node:20-alpine
WORKDIR /test
RUN apk add --no-cache cargo rust
COPY package.json .
COPY *.node .
COPY index.js .
RUN npm install
CMD ["node", "-e", "console.log(require('./index.js'))"]
