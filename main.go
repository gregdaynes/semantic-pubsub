// proof-of-concept semantic pub/sub.
//
// GET /sub?query=... embeds the query with amazon/titan-embed-text-v2 on
// OpenRouter and holds an SSE connection.
// POST /publish {"message": "..."} embeds the message and streams its text
// to every subscriber whose query matches: cosine similarity >= threshold
// OR a query keyword appears literally.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"sync"
	"syscall"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
)

const modelID = "amazon/titan-embed-text-v2"

type subscriber struct {
	keywords []string
	vec      []float32
	messages chan string
}

type server struct {
	g         *genkit.Genkit
	embedder  ai.Embedder
	threshold float64

	mu   sync.Mutex
	subs map[*subscriber]bool
}

func (s *server) embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := genkit.Embed(ctx, s.g,
		ai.WithEmbedder(s.embedder),
		ai.WithTextDocs(text),
	)
	if err != nil {
		return nil, err
	}
	return resp.Embeddings[0].Embedding, nil
}

func (s *server) subscribe(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("query")
	if query == "" {
		http.Error(w, "missing query parameter", http.StatusBadRequest)
		return
	}

	vec, err := s.embed(r.Context(), query)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	sub := &subscriber{keywords: keywords(query), vec: vec, messages: make(chan string, 16)}
	s.mu.Lock()
	s.subs[sub] = true
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.subs, sub)
		s.mu.Unlock()
	}()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	fmt.Fprint(w, ": subscribed\n\n")
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			return
		case msg := <-sub.messages:
			fmt.Fprintf(w, "data: %s\n\n", msg)
			flusher.Flush()
		}
	}
}

func (s *server) publish(w http.ResponseWriter, r *http.Request) {
	var body struct {
		Message string `json:"message"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Message == "" {
		http.Error(w, `expected JSON {"message": "..."}`, http.StatusBadRequest)
		return
	}

	vec, err := s.embed(r.Context(), body.Message)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	text := strings.ToLower(body.Message)
	delivered := 0

	s.mu.Lock()
	for sub := range s.subs {
		if cosine(vec, sub.vec) >= s.threshold || containsAny(text, sub.keywords) {
			select {
			case sub.messages <- body.Message:
				delivered++
			default: // subscriber buffer full: drop rather than block the publisher
			}
		}
	}
	s.mu.Unlock()

	fmt.Fprintf(w, "delivered to %d subscriber(s)\n", delivered)
}

// this is lazy, a porter-stem approach or other fts would be better
func keywords(query string) []string {
	return strings.Fields(strings.ReplaceAll(strings.ToLower(query), ",", " "))
}

func containsAny(text string, keywords []string) bool {
	for _, k := range keywords {
		if strings.Contains(text, k) {
			return true
		}
	}

	return false
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64

	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}

	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func main() {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENROUTER_API_KEY is required")
	}
	baseURL := os.Getenv("OPENROUTER_BASE_URL")
	if baseURL == "" {
		baseURL = "https://openrouter.ai/api/v1"
	}
	threshold := 0.5
	if v := os.Getenv("MATCH_THRESHOLD"); v != "" {
		t, err := strconv.ParseFloat(v, 64)
		if err != nil {
			log.Fatalf("invalid MATCH_THRESHOLD: %v", err)
		}
		threshold = t
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	plugin := &compat_oai.OpenAICompatible{
		Provider: "openrouter",
		APIKey:   apiKey,
		BaseURL:  baseURL,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	s := &server{
		g:         g,
		embedder:  plugin.NewEmbedder(modelID, nil),
		threshold: threshold,
		subs:      make(map[*subscriber]bool),
	}

	http.HandleFunc("GET /sub", s.subscribe)
	http.HandleFunc("POST /publish", s.publish)
	srv := &http.Server{Addr: ":" + port}

	go func() {
		<-ctx.Done()
		srv.Close() // ungraceful shutdown because sse and I'm lazy
	}()

	log.Printf("listening on :%s (threshold %.2f)", port, threshold)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}
