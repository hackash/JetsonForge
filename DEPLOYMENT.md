# Docker Registry Deployment Examples

This document provides complete examples for deploying Jetson Docker images to different container registries.

---

## Prerequisites

All examples require a **multi-stage Dockerfile**:

```dockerfile
ARG L4T_CROSS_BASE=jetson-cross-base:jp6.0-orin-nano
ARG TARGET_JETPACK_TAG=r36.3.0

# Stage 1: Cross-compile on x86_64
FROM ${L4T_CROSS_BASE} AS builder
WORKDIR /app
COPY . .
RUN cmake -B build -S . \
      -DCMAKE_TOOLCHAIN_FILE=/toolchains/aarch64-jetson.cmake \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j$(nproc)

# Stage 2: Runtime for Jetson ARM64
FROM nvcr.io/nvidia/l4t-jetpack:${TARGET_JETPACK_TAG}
COPY --from=builder /app/build/my-app /usr/local/bin/my-app
CMD ["/usr/local/bin/my-app"]
```

See [examples/cmake-cuda/Dockerfile.production](../examples/cmake-cuda/Dockerfile.production) for a complete example.

---

## 1. GitHub Container Registry (GHCR)

**Advantages:** Free, integrated with GitHub, no external accounts needed.

```yaml
name: Deploy to GHCR

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Cross-Compiler
        id: jetsonforge
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
          jetson_target: 'JETSON_ORIN_NANO_TARGETS'
      
      - name: Build Production Image
        run: |
          docker build \
            --build-arg L4T_CROSS_BASE=${{ steps.jetsonforge.outputs.docker_image }} \
            --build-arg TARGET_JETPACK_TAG=r36.3.0 \
            -t my-jetson-app:${{ github.sha }} \
            .
      
      - name: Login to GHCR
        run: |
          echo ${{ secrets.GITHUB_TOKEN }} | \
            docker login ghcr.io -u ${{ github.actor }} --password-stdin
      
      - name: Push to GHCR
        run: |
          IMAGE=ghcr.io/${{ github.repository_owner }}/my-jetson-app
          docker tag my-jetson-app:${{ github.sha }} $IMAGE:${{ github.sha }}
          docker tag my-jetson-app:${{ github.sha }} $IMAGE:latest
          docker push $IMAGE:${{ github.sha }}
          docker push $IMAGE:latest
```

**Deploy to Jetson:**
```bash
# On your Jetson device
docker login ghcr.io -u YOUR_USERNAME
docker pull ghcr.io/YOUR_USERNAME/my-jetson-app:latest
docker run -d --runtime nvidia ghcr.io/YOUR_USERNAME/my-jetson-app:latest
```

---

## 2. AWS Elastic Container Registry (ECR)

**Advantages:** Integrated with AWS ecosystem, good for IoT/edge fleets.

```yaml
name: Deploy to AWS ECR

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Cross-Compiler
        id: jetsonforge
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
          jetson_target: 'JETSON_ORIN_NANO_TARGETS'
      
      - name: Build Production Image
        run: |
          docker build \
            --build-arg L4T_CROSS_BASE=${{ steps.jetsonforge.outputs.docker_image }} \
            --build-arg TARGET_JETPACK_TAG=r36.3.0 \
            -t my-jetson-app:${{ github.sha }} \
            .
      
      - name: Configure AWS Credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2
      
      - name: Push to ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: my-jetson-app
        run: |
          docker tag my-jetson-app:${{ github.sha }} \
            $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}
          docker tag my-jetson-app:${{ github.sha }} \
            $ECR_REGISTRY/$ECR_REPOSITORY:latest
          
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:${{ github.sha }}
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

**Deploy to Jetson:**
```bash
# Install AWS CLI on Jetson
sudo apt-get install awscli

# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# Pull and run
docker pull 123456789.dkr.ecr.us-east-1.amazonaws.com/my-jetson-app:latest
docker run -d --runtime nvidia \
  123456789.dkr.ecr.us-east-1.amazonaws.com/my-jetson-app:latest
```

---

## 3. Docker Hub

**Advantages:** Most widely used, simple, good for public images.

```yaml
name: Deploy to Docker Hub

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Cross-Compiler
        id: jetsonforge
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
          jetson_target: 'JETSON_ORIN_NANO_TARGETS'
      
      - name: Build Production Image
        run: |
          docker build \
            --build-arg L4T_CROSS_BASE=${{ steps.jetsonforge.outputs.docker_image }} \
            --build-arg TARGET_JETPACK_TAG=r36.3.0 \
            -t my-jetson-app:${{ github.sha }} \
            .
      
      - name: Login to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}
      
      - name: Push to Docker Hub
        run: |
          REPO=${{ secrets.DOCKERHUB_USERNAME }}/my-jetson-app
          docker tag my-jetson-app:${{ github.sha }} $REPO:${{ github.sha }}
          docker tag my-jetson-app:${{ github.sha }} $REPO:latest
          docker push $REPO:${{ github.sha }}
          docker push $REPO:latest
```

**Deploy to Jetson:**
```bash
# On your Jetson device (public images don't need login)
docker pull myusername/my-jetson-app:latest
docker run -d --runtime nvidia myusername/my-jetson-app:latest
```

---

## 4. Google Artifact Registry

**Advantages:** Integrated with GCP, good for Google Cloud IoT deployments.

```yaml
name: Deploy to Google Artifact Registry

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Cross-Compiler
        id: jetsonforge
        uses: hackash/JetsonForge@main
        with:
          jetpack_version: '6.0'
          jetson_target: 'JETSON_ORIN_NANO_TARGETS'
      
      - name: Build Production Image
        run: |
          docker build \
            --build-arg L4T_CROSS_BASE=${{ steps.jetsonforge.outputs.docker_image }} \
            --build-arg TARGET_JETPACK_TAG=r36.3.0 \
            -t my-jetson-app:${{ github.sha }} \
            .
      
      - name: Authenticate to Google Cloud
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_CREDENTIALS }}
      
      - name: Setup Cloud SDK
        uses: google-github-actions/setup-gcloud@v2
      
      - name: Configure Docker for Artifact Registry
        run: gcloud auth configure-docker us-central1-docker.pkg.dev
      
      - name: Push to Artifact Registry
        env:
          PROJECT_ID: my-gcp-project
          REGION: us-central1
          REPOSITORY: jetson-apps
        run: |
          IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/my-jetson-app
          docker tag my-jetson-app:${{ github.sha }} $IMAGE:${{ github.sha }}
          docker tag my-jetson-app:${{ github.sha }} $IMAGE:latest
          docker push $IMAGE:${{ github.sha }}
          docker push $IMAGE:latest
```

**Deploy to Jetson:**
```bash
# Install gcloud SDK on Jetson
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate
gcloud auth login
gcloud auth configure-docker us-central1-docker.pkg.dev

# Pull and run
docker pull us-central1-docker.pkg.dev/my-project/jetson-apps/my-jetson-app:latest
docker run -d --runtime nvidia \
  us-central1-docker.pkg.dev/my-project/jetson-apps/my-jetson-app:latest
```

---

## Fleet Deployment with Docker Compose

For managing multiple containers on Jetson:

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  my-app:
    image: ghcr.io/my-org/my-jetson-app:latest
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=all
    volumes:
      - /data:/data
    network_mode: host

  monitoring:
    image: prom/prometheus:latest
    restart: unless-stopped
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
```

**Deploy:**
```bash
# On Jetson
docker-compose pull
docker-compose up -d
```

---

## Automated Jetson Updates

**Update script for Jetson devices:**
```bash
#!/bin/bash
# update-app.sh - Run on Jetson device

IMAGE="ghcr.io/my-org/my-jetson-app:latest"
CONTAINER="my-app"

echo "Pulling latest image..."
docker pull $IMAGE

echo "Stopping current container..."
docker stop $CONTAINER || true
docker rm $CONTAINER || true

echo "Starting new container..."
docker run -d \
  --name $CONTAINER \
  --runtime nvidia \
  --restart unless-stopped \
  $IMAGE

echo "Cleanup old images..."
docker image prune -f

echo "Update complete!"
docker ps | grep $CONTAINER
```

---

## Security Best Practices

1. **Use specific tags** instead of `latest` in production
2. **Scan images** for vulnerabilities before deployment
3. **Use secrets** for registry credentials, never commit them
4. **Enable image signing** (Docker Content Trust, Cosign)
5. **Run as non-root** user inside containers
6. **Limit container capabilities** and resources

---

## See Also

- [Main Documentation](README-GHA.md)
- [Example Dockerfile](../examples/cmake-cuda/Dockerfile.production)
- [JetPack Version Guide](JETPACK-VERSIONS.md)
