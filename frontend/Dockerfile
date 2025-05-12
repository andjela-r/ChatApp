FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
COPY next.config.ts ./next.config.ts

EXPOSE 3001

CMD ["npm", "run", "dev", "--", "-p", "3001"]