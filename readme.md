A proof-of-concept semantic pub/sub.

GET /sub?query=... embeds the query with amazon/titan-embed-text-v2 on OpenRouter and holds an SSE connection.

POST /publish {"message": "..."} embeds the message and streams its text to every subscriber whose query matches: cosine similarity >= threshold OR a query keyword appears literally.


## Run

1. Copy .env.sample to .env and populate OPENROUTER_API_KEY=

2. Run application in a terminal

```sh
export $(cat .env) && go run .
```

3. In another terminal create a subscription

```sh
curl -N 'http://localhost:4000/sub?query=climate%20change'
```

4. In a third terminal send a message to be matched against

```sh
curl -X POST http://localhost:4000/publish -H 'Content-Type: application/json' -d '{"message":"Weather systems are undergoing profound changes as the planet warms. Jet streams are slowing and meandering, letting heat domes and cold snaps stall over the same region for weeks. Hurricanes are tracking farther north, rain arrives in shorter and more violent bursts, and the dry spells between storms stretch longer. The seasonal rhythms that agriculture and city planning depended on for a century are becoming unreliable."}'
```

You should see the subscription terminal print out the message based on a match of the query and the content based on cosine similarity.
