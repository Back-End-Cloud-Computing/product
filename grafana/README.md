# Dashboard Grafana — `product`

[`product-dashboard.json`](product-dashboard.json): requisições, erros e latência do product-service, a partir do
`/metrics` (`prometheus-fastapi-instrumentator`).

| Painel | O que mostra |
|---|---|
| Requisições/s, Erros 5xx, Latência p95, Réplicas no ar | Resumo em números |
| Requisições/s por rota | Tráfego por `método + rota` |
| Respostas por status | `2xx`, `4xx` (inclui 401 sem token), `5xx` |
| Latência (p50, p95, p99) e p95 por rota | `POST /products` deve ficar bem abaixo dos 10 s do gateway |
| Taxa de erro 5xx por rota | Qual rota está falhando |

## Importar

Grafana → **Dashboards → New → Import** → enviar o JSON. Não precisa editar nada:

- **Prometheus** (variável `datasource`): escolha o datasource no topo do painel.
- **Filtro** (variável `filtro`, padrão `app="product-api"`): seletor que identifica os pods do product. É o label
  que o Prometheus cria a partir do label `app` do pod; se o seu scrape usar outro, troque no campo do topo.

Validado em 24/10/2026 com Prometheus 3.5 e Grafana 11.6 raspando o chart instalado num `kind`: import aceito e todas
as consultas dos painéis retornando dados.
