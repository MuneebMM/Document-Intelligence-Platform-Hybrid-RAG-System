# =============================================================================
# Stage 1 — builder
# Install dependencies and compile the React/Vite application.
# =============================================================================
FROM node:20-alpine AS builder

WORKDIR /app

# Copy lockfiles first so Docker cache is invalidated only when dependencies
# change, not on every source edit.
COPY package.json package-lock.json ./

# npm ci is faster than npm install, uses the lockfile exactly, and fails on
# any mismatch — the right choice for repeatable CI/CD builds.
RUN npm ci

# Copy the rest of the source after installing deps.
COPY . .

# Produce the optimised static bundle in /app/dist.
RUN npm run build


# =============================================================================
# Stage 2 — production
# Minimal nginx image: no Node, no npm, just compiled static files and nginx.
# =============================================================================
FROM nginx:alpine

# Custom nginx config — replaces the default site so our rules take effect.
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Static assets compiled in Stage 1.
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
