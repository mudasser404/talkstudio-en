# Docker Hub Setup Instructions

## Problem: GitHub Actions Docker Login Failed

Agar "Log in to Docker Hub" step fail ho raha hai, to ye steps follow karo:

## Step 1: Docker Hub Access Token Banao

1. **Docker Hub** pe login karo: https://hub.docker.com/
2. Right top corner me **Account Settings** (profile icon) pe click karo
3. **Security** tab pe jao
4. **New Access Token** button click karo
5. Token description dalo (e.g., "GitHub Actions")
6. **Access Permissions**: Select "Read, Write, Delete"
7. **Generate** button click karo
8. **Token ko copy kar lo** (ye sirf ek baar dikhega!)

## Step 2: GitHub Repository Secrets Add Karo

1. GitHub repository pe jao: https://github.com/mudasser404/talkstudio-en
2. **Settings** tab click karo (repo ka, apna profile ka nahi!)
3. Left sidebar me **Secrets and variables** > **Actions** pe jao
4. **New repository secret** button click karo

### Add karo ye 2 secrets:

**Secret 1: DOCKER_USERNAME**
- Name: `DOCKER_USERNAME`
- Value: Tumhara Docker Hub username (NOT email)
- **Add secret** click karo

**Secret 2: DOCKER_PASSWORD**
- Name: `DOCKER_PASSWORD`
- Value: Docker Hub access token (jo Step 1 me copy kiya)
- **Add secret** click karo

## Step 3: Workflow Manually Trigger Karo

1. GitHub repository me **Actions** tab pe jao
2. Left sidebar me **Build and Push Docker Image** workflow select karo
3. Right side me **Run workflow** dropdown click karo
4. Branch select karo: `main`
5. **Run workflow** green button click karo

## Step 4: Verify

- Workflow run hogi (15-20 minutes lagenge)
- Sab steps green ho jayengi ✓
- Docker Hub pe image dikhai degi: `your-username/f5-tts:latest`

## Common Issues

### Issue 1: "unauthorized: incorrect username or password"
- **Fix**: DOCKER_USERNAME me email use kar rahe ho. Username use karo, email nahi!
- Docker Hub username check karo: https://hub.docker.com/settings/general

### Issue 2: "denied: requested access to the resource is denied"
- **Fix**: Access token ke permissions check karo. "Read, Write, Delete" hone chahiye.
- Naya token banao proper permissions ke sath.

### Issue 3: Secrets dikh nahi rahe
- **Fix**: Repo ke **Settings** me ho, profile settings me nahi!
- URL should be: `https://github.com/mudasser404/talkstudio-en/settings/secrets/actions`

## Verification Commands

Agar manually test karna ho:

```bash
# Docker login test
echo "YOUR_ACCESS_TOKEN" | docker login -u YOUR_USERNAME --password-stdin

# Build test
docker build -t your-username/f5-tts:test .

# Push test
docker push your-username/f5-tts:test
```
