FROM node:18-alpine

WORKDIR /app

# Simple static file server
RUN npm install -g http-server

# Copy frontend files
COPY . /app/frontend

# Expose the server port
EXPOSE 3000

# Start the static server
CMD ["http-server", "frontend", "-p", "3000", "-c", "-1"]
