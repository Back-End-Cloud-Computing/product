# Chart Helm — `product`

Mesmo desenho do [`k8s/`](../../k8s/README.md): Deployment `product-api` (3 réplicas, anotações do Prometheus),
Service `product-api:8000`, ConfigMap, Ingress por lista de permissão (`/products`, `/search`, `/docs`,
`/openapi.json`, `/health`) e MongoDB opcional. Os valores que mudam entre ambientes estão em
[`values.yaml`](values.yaml).

O Secret continua fora do chart (nenhum `kind: Secret` no git). Crie `product-secrets` como no
[`k8s/README.md`](../../k8s/README.md) e passe outro nome com `--set existingSecret=...` se precisar.
O product busca a chave pública do `authorization` ao subir, então instale-o antes.

```bash
kubectl create secret generic product-secrets --from-literal=INTERNAL_SECRET='<segredo>'
helm lint helm/product
helm upgrade --install product helm/product
kubectl get pods -w
helm uninstall product
```

Ajustes comuns:

| Quero | Faço |
|---|---|
| Outra versão da imagem | `--set image.tag=1.1.1` |
| Mais réplicas | `--set replicaCount=5` |
| Outro host do Ingress | `--set ingress.host=product.exemplo.local` |
| Banco externo | `--set mongodb.enabled=false --set config.MONGODB_URI=<uri>` |
| Sem Ingress | `--set ingress.enabled=false` |

Testado em 24/10/2026 num cluster `kind` limpo (Kubernetes 1.37, Helm 4.3): `helm lint` limpo, `helm install` com o
`values.yaml` padrão sobe 3 pods `product-api` `1/1` e o MongoDB. Pelo Service `product-api:8000`, `/health` e
`/metrics` respondem 200, `/products` sem token 401 e `/internal/...` com o `INTERNAL_SECRET` do Secret chega ao banco.
