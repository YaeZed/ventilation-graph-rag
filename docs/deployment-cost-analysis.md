# Deployment Cost Analysis

Updated: 2026-06-03

## Cost Summary

For this project, deployment cost has two different profiles:

- If public deployment uses `BYOK local-only`, the maintainer's fixed cost can stay low. Users pay their own model API bills.
- If the maintainer provides the model key, model usage becomes the main risk. Image and multi-image requests can outgrow server cost quickly.

The recommended student/beta budget is:

| Tier | Monthly Cost | What It Covers | Main Risk |
|---|---:|---|---|
| Minimal public beta | 50-150 RMB/month, if using discounted VPS and free graph/vector tiers | 4C8G-ish VPS/light server, local Postgres, nginx, Vue static, media volume, AuraDB Free, Zilliz Free, BYOK | Free cloud tiers may hit limits; VPS promo renewal may rise |
| Safer beta | 150-400 RMB/month | Better VPS/disk/backups, still external free/low-cost graph/vector, BYOK | Still needs ops discipline and import-size validation |
| Maintainer-paid model demo | fixed cost + 200-2,000+ RMB/month model usage | Same infra, but maintainer pays DashScope/OpenAI calls | Unbounded user usage burns quota |
| Paid Neo4j production | fixed cost + at least $65.70/month for AuraDB Professional 1GB | Managed production graph database | Neo4j becomes larger than the VPS bill |
| Self-host all-in-one | 8C16G+ server cost, often several hundred to 1,000+ RMB/month depending provider/discount | VPS hosts Django, Postgres, Neo4j, Milvus, nginx, media | Higher ops burden and less stable than externalizing vector/graph |

## Fixed Infrastructure Cost

| Item | Recommended First Choice | Expected Cost | Notes |
|---|---|---:|---|
| VPS/ECS | 4C8G, annual discount if available | 50-150 RMB/month discounted; standard on-demand can be much higher | 2C4G is tight because Django loads torch/BGE and handles long SSE/image requests. |
| Postgres | Container on the same VPS | 0 incremental | Backups still cost time/storage. Managed Postgres is cleaner but adds monthly cost. |
| nginx + Vue static | Same VPS | 0 incremental | Same-origin routing avoids extra CSRF/CORS work. |
| Media files | VPS persistent volume first | Included until disk grows | Needs backup/retention policy. |
| Domain | Normal registrar | ~30-100 RMB/year | Optional for a private IP-only demo, required for normal HTTPS UX. |
| TLS | Let's Encrypt | 0 | Operational work only. |
| Object storage | Later, Cloudflare R2 or cloud OSS/COS | R2 has 10GB-month free, then $0.015/GB-month standard storage | Not required for first deployment if media volume is backed up. |

## Knowledge Store Cost

| Component | Recommended First Choice | Cost | Cost Trigger |
|---|---|---:|---|
| Neo4j | AuraDB Free first | $0 | Free plan has node/relationship limits and no production SLA/backup guarantees. |
| Neo4j paid | AuraDB Professional | starts at $65.70/month for 1GB cluster | This is the first major fixed-cost jump. |
| Vector DB | Zilliz Cloud Free | $0 | Free cluster includes 5GB storage, 2.5M vCUs/month, up to 5 collections. |
| Self-hosted Milvus | Avoid on small VPS | Requires bigger server | Official standalone requirement: 4+ CPU cores, 8GB RAM minimum, 16GB recommended. Current local compose also needs etcd and MinIO. |

## Model API Cost

### Recommended Policy

Use `BYOK local-only` for public deployment:

- User API keys stay in the browser/local request path.
- Keys are sent only with chat/model-test requests.
- Keys must not persist in `UserProfile.settings`.

This makes the maintainer model bill close to 0 for public users. The cost moves to each user, which is the correct boundary for a student project unless this is a private demo.

### If The Maintainer Pays

DashScope prices checked from Alibaba Cloud Model Studio pricing:

| Model | Price Basis | Cost Signal |
|---|---|---|
| `qwen-plus` China mainland | input 0.8 RMB / 1M tokens, output 2 RMB / 1M tokens for <=128K non-thinking requests | Text chat is cheap. |
| `qwen3.5-plus` China mainland | input 0.8 RMB / 1M tokens, output 4.8 RMB / 1M tokens for <=128K | Stronger text model, still manageable. |
| `qwen3.5-omni-plus` China mainland | text/image/video input 7 RMB / 1M tokens, text output 40 RMB / 1M tokens | Image requests are the real cost driver. |
| `qwen3.5-omni-flash` China mainland | text/image/video input 2.2 RMB / 1M tokens, text output 13.3 RMB / 1M tokens | Cheaper image fallback if quality is acceptable. |

Rough per-request estimates for this project's RAG flow:

| Request Type | Assumption | Rough Cost With DashScope |
|---|---|---:|
| Text-only question | 5K-20K input tokens + 1K-2K output tokens | 0.01-0.05 RMB |
| Single-image recognition | 2 Omni calls + final text generation | 0.3-1.0 RMB |
| 3-image fused recognition | per-image observations + joint analysis + final text generation | 0.8-3.0 RMB |

These are rough because image tokens depend on resolution and provider-side tokenization. The safest product control is a monthly budget cap plus rate limits.

Example monthly model spend if the maintainer pays:

| Usage | Approx Monthly Model Cost |
|---|---:|
| 30 text-only requests/day | 10-45 RMB |
| 10 single-image requests/day | 90-300 RMB |
| 50 single-image requests/day | 450-1,500 RMB |
| 20 three-image requests/day | 480-1,800 RMB |

## Cost Decisions

| Decision | Recommendation | Why |
|---|---|---|
| First public beta key policy | BYOK local-only | Removes the largest unpredictable cost. |
| Default user-facing image model | Keep Plus for accuracy first; evaluate Flash after deployment | Safety/accuracy matters for mine-ventilation analysis. Cost optimization should not silently degrade output quality. |
| Free graph/vector tiers | Use first, validate import size | Keeps fixed cost low. |
| Paid Neo4j | Defer until AuraDB Free is proven insufficient | $65.70/month is a large jump for a student project. |
| Object storage | Defer until media volume/backups become painful | R2 is cheap, but current code already works with local media. |
| All-in-one self-hosting | Avoid unless using 8C16G+ | Milvus and Neo4j will make small servers unstable. |

## Practical Budget Recommendation

Start with this:

```text
4C8G discounted VPS/light server
Postgres in Docker
nginx + Vue dist + Django/Gunicorn
media persistent volume
Neo4j AuraDB Free
Zilliz Cloud Free
BYOK local-only
```

Expected maintainer cost:

- Fixed: about 50-150 RMB/month if discounted server pricing is available; otherwise plan for several hundred RMB/month.
- Model: near 0 for public users under BYOK.
- Upgrade trigger: if AuraDB Free cannot hold the graph, add at least $65.70/month or self-host Neo4j on a larger server.

## Sources

- Alibaba Cloud Model Studio pricing: https://help.aliyun.com/zh/model-studio/model-pricing
- Neo4j pricing: https://neo4j.com/pricing/
- Zilliz free cluster/trial: https://docs.zilliz.com/docs/free-trials
- Milvus standalone requirements: https://milvus.io/docs/prerequisite-docker.md
- Cloudflare R2 pricing: https://developers.cloudflare.com/r2/pricing/
- OpenAI API pricing: https://openai.com/api/pricing/
