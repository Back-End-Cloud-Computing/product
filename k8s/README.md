# product-service no Kubernetes

Sobe o Mongo e três réplicas do product-service, com Service Discovery entre
eles e as dependências do GANJJ (`authorization`, `vector-db`,
`embedding-reranking`, `llm-provider`).

```
Service product-api :8000
          |
   Service Discovery
          |
   +------+------+------+
   |             |      |
  Pod           Pod    Pod
   |             |      |
   +------+------+------+
          |
          | mongodb:27017, authorization:8081, vector-db:8002,
          | embedding-reranking:8003, llm-provider:8004
          v
   Services correspondentes (cada um no k8s/ do seu proprio repositorio)
```

## Subir

Este serviço depende de `authorization`, `vector-db`, `embedding-reranking` e
`llm-provider` já estarem no cluster (cada um com seu próprio `k8s/`, no seu
próprio repositório):

```bash
kubectl apply -f k8s/mongodb.yaml
kubectl apply -f k8s/product.yaml
```

```bash
kubectl wait --for=condition=ready pod -l app=mongodb --timeout=120s
kubectl wait --for=condition=ready pod -l app=product-api --timeout=120s
kubectl get pods
```

## Testar a comunicação interna

```bash
kubectl run teste --image=curlimages/curl:latest -it --rm -- sh
```

```sh
# sem token: 401
curl -s -o /dev/null -w "%{http_code}\n" http://product-api:8000/products

# login no authorization (mesmo cluster, Service "authorization")
T=$(curl -s -X POST http://authorization:8081/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@ganjj.com","password":"adminSegura123"}' \
  | sed 's/.*"accessToken":"\([^"]*\)".*/\1/')

# com token: 200 - e por trás dessa chamada o product-api já fala com
# mongodb, vector-db, embedding-reranking e llm-provider, todos pelo nome do
# Service
curl -s -o /dev/null -w "%{http_code}\n" http://product-api:8000/products \
  -H "Authorization: Bearer $T"
```

## Ver a distribuição entre as réplicas

```sh
for i in $(seq 1 10); do
  curl -s -o /dev/null -w "%{http_code}\n" http://product-api:8000/health
done
```

## Autorrecuperação

```bash
kubectl get pods -l app=product-api
kubectl delete pod <nome-de-um-pod>
kubectl get pods -l app=product-api
```

O manifesto declara `replicas: 3`. Ao perder um Pod, o Deployment cria outro
para voltar ao estado declarado. O Service atualiza os endpoints sozinho, e
quem consome continua usando o mesmo endereço `product-api:8000`.

## Escalar

Altere `replicas` em `product.yaml` e reaplique. Não use `kubectl scale`: o
arquivo no Git deve representar o estado desejado.

```bash
kubectl apply -f k8s/product.yaml
```

## Acessar do host

```bash
kubectl port-forward service/product-api 8000:8000
```

## Acessar via Ingress (sem port-forward por serviço)

```bash
minikube addons enable ingress
kubectl apply -f k8s/ingress.yaml
kubectl wait --namespace ingress-nginx --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller --timeout=120s
```

Testar pelo próprio NGINX Ingress Controller (host `product.ganjj.local`, sem
DNS nenhum configurado — só o header `Host`):

```bash
IP=$(minikube ip)
PORT=$(kubectl get svc -n ingress-nginx ingress-nginx-controller -o jsonpath='{.spec.ports[?(@.port==80)].nodePort}')
curl -H "Host: product.ganjj.local" http://$IP:$PORT/docs
```

**No WSL2** (driver docker roda numa rede Docker interna à VM, que o Windows
não alcança direto): encaminhe **uma porta só, para o Ingress Controller**
— não para o `product-api` diretamente. Quem roteia continua sendo o Ingress,
pelo `Host`:

```bash
kubectl port-forward -n ingress-nginx service/ingress-nginx-controller 8080:80
```

E no hosts do **Windows** (`C:\Windows\System32\drivers\etc\hosts`, como
administrador): `127.0.0.1  product.ganjj.local`. Depois, no navegador:
http://product.ganjj.local:8080/docs

## Sobre a imagem

Publicada em [joao2006/product](https://hub.docker.com/r/joao2006/product).
`imagePullPolicy: IfNotPresent` evita baixar de novo quando o cluster já tem a
tag em cache local, mas a imagem existe no Docker Hub, então o manifesto sobe
em qualquer cluster.

Para publicar uma versão nova:

```bash
docker build -t joao2006/product:1.0.1 .
docker push joao2006/product:1.0.1
```

E atualize a tag em `image:` no `product.yaml`.

## Limitação conhecida

O Mongo roda como Deployment sem volume persistente: se o Pod for recriado, o
banco começa vazio. Para o laboratório basta. Um banco de verdade pediria
StatefulSet com PersistentVolumeClaim.
