# Render Deployment

Production deployment of the FastAPI service on Render's free tier.

## Files

- [render.yaml](../render.yaml) — Blueprint (services, build/start commands, env vars, health check).
- [requirements.txt](../requirements.txt) — Locked Python deps used by the build step.
- [app/main.py](../app/main.py) — ASGI entrypoint exposing `app` (used as `app.main:app`).

## One-time setup

1. Push this repo to GitHub.
2. In Render: **New +** → **Blueprint** → connect the repo. Render reads `render.yaml` and provisions the `product-review-intelligence` web service.
3. Open the service → **Environment** and fill in every var marked `sync: false` (secrets):
   - `API_KEY`, `PIPELINE_API_KEY`
   - `GROQ_API_KEY`
   - `MCP_SERVER_URL`, `MCP_TOKEN`
   - `PLAYSTORE_APP_ID`, `APPSTORE_APP_ID`
   - `EMAIL_TO`
4. Save — Render redeploys automatically.

Pushes to `main` auto-deploy (`autoDeploy: true`).

## Runtime

- **Python:** 3.11.9 (pinned via `PYTHON_VERSION`)
- **Port:** Render injects `$PORT`; uvicorn binds to it.
- **Workers:** Single async worker (`WEB_CONCURRENCY=1`). Free tier has 512 MB RAM; one async worker is the right shape for an I/O-bound app and avoids OOM kills.
- **No Docker:** Deploys as a standard Render Python web service. Render runs `buildCommand` then `startCommand` directly against the Python runtime — no container build step.

### Start command

```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### Build command

```
pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
```

`--no-cache-dir` shaves disk on the free build container.

## Health checks

Render polls `healthCheckPath: /healthz` and only routes traffic once it returns 200. Endpoints exposed by [app/api/routes/health.py](../app/api/routes/health.py):

| Path       | Purpose                              |
| ---------- | ------------------------------------ |
| `/healthz` | Render liveness probe (cheap, fast). |
| `/readyz`  | Readiness — extend if deps need a warm check. |
| `/health`  | Verbose: app name + environment.     |

## Free-tier notes

- The instance sleeps after ~15 min idle and cold-starts on the next request (~30–60 s). For scheduled work, drive it from GitHub Actions (already wired) so the cron lives outside Render.
- 512 MB RAM ceiling: `MAX_REVIEWS_PER_RUN` and `LLM_MAX_CONCURRENCY` are dialled down in `render.yaml` vs. local defaults. Raise them only after observing the **Metrics** tab.
- No persistent disk on free web services — write only to `/tmp` if you need scratch space.

## Verifying a deploy

```
curl https://<your-service>.onrender.com/healthz
# {"status":"ok"}

curl https://<your-service>.onrender.com/health
# {"status":"ok","app":"Product-Review-Intelligence","environment":"production"}
```

If `/healthz` fails during boot, check **Logs** for the uvicorn startup line and the `app.startup` structlog event.
