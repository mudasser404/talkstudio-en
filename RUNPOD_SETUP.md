# RunPod Deployment Setup

## Step 1: Docker Hub Setup

1. **Docker Hub account banao** (agar nahi hai): https://hub.docker.com/
2. **Access Token generate karo**:
   - Docker Hub pe login karo
   - Account Settings > Security > New Access Token
   - Token copy kar lo

## Step 2: GitHub Secrets Setup

1. GitHub repository pe jao: https://github.com/mudasser404/talkstudio-en
2. **Settings** > **Secrets and variables** > **Actions**
3. Do secrets add karo:
   - `DOCKER_USERNAME`: Tumhara Docker Hub username
   - `DOCKER_PASSWORD`: Docker Hub access token (jo step 1 me banaya)

## Step 3: Workflow Trigger

Ab jab bhi tum `Dockerfile` ko push karoge, automatically:
- GitHub Actions image build karega
- Docker Hub pe push karega
- Tag hoga: `your-username/f5-tts:latest`

## Step 4: RunPod Deployment

1. **RunPod** pe login karo: https://www.runpod.io/
2. **Deploy** > **Custom Container**
3. **Container Image** me dalo:
   ```
   your-docker-username/f5-tts:latest
   ```
4. **Expose HTTP Ports** me dalo: `7860`
5. **Deploy** karo

## Step 5: Access Gradio Interface

- RunPod dashboard me **Connect** button pe click karo
- **Connect to HTTP Service [7860]** select karo
- Gradio interface khul jayega!

## Manual Build (Optional)

Agar manually build karna ho:

```bash
docker build -t your-username/f5-tts:latest .
docker push your-username/f5-tts:latest
```

## Troubleshooting

- **Build fail ho raha hai**: GitHub Actions logs check karo
- **RunPod pe start nahi ho raha**: Container logs dekho RunPod dashboard me
- **Models download nahi ho rahe**: Internet access check karo container me
