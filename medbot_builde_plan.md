## 1. Vue d’ensemble : macro-architecture cible

```
┌───────────────────────────────┐
│    Terminal patient (mobile)  │  Flutter / SwiftUI / Jetpack →  TLS 1.3
│  • Chat           • Caméra    │––––––––––––––––––––––––––––┐
│  • Capteurs BLE   • Stockage  │                          API Gateway (REST+gRPC)
└───────────────────────────────┘                          │
          ▲  ▲                                             │ OAuth2 / OIDC
          │  │  BLE / USB                                   ▼
┌─────────┴──┴──────────┐        ┌──────────────────────────────────────────────┐
│  Dispositif biosignaux│        │     Plan de contrôle (orchestrateur)         │
│  (µC Zephyr, BLE, …)  │───────▶│ • Construction du prompt RAG                 │
│  ECG, PPG, BP…        │        │ • Gestion contexte patient & historique      │
└───────────────────────┘        │ • Routage vers le LLM                         │
                                 └─────────────┬────────────────────────────────┘
                                               │ gRPC streaming / Batched
                                 ┌─────────────▼───────────────────────┐
                                 │  Service d’inférence MedGemma 4B    │
                                 │  (GPU A100 / H100, Triton/TF-Serving)│
                                 └─────────────┬───────────────────────┘
                                               │
                                 ┌─────────────▼────────────────────────┐
                                 │  Services métiers sécurisés          │
                                 │ • FHIR Store (DiagnosticReport,      │
                                 │   Observation, MedicationRequest)    │
                                 │ • Vector-DB (FAISS / PG-pgvector)    │
                                 │ • MinIO chiffrement at-rest          │
                                 └─────────────┬────────────────────────┘
                                               │ WebSocket / FHIR-Sub
┌───────────────────────────────┐              ▼
│  Portail clinicien (Web)      │  Revue-Approbation ordonnance + audit
└───────────────────────────────┘
```

## 2. Périmètre fonctionnel

| Mode                     | Fonction patient                    | Fonction clinicien                           | Sortie normalisée                        |
| ------------------------ | ----------------------------------- | -------------------------------------------- | ---------------------------------------- |
| **Dialogue**             | Questions, images, mesures capteurs | Suivi conversation                           | `Bundle` FHIR (messages)                 |
| **Rapport & ordonnance** | Questionnaire guidé, uploads        | Validation, signature électronique qualifiée | `DiagnosticReport` + `MedicationRequest` |

L’orchestreur fixe la *system prompt* pour MedGemma 4B, y injecte :

- l’historique,
- les passages de documents cliniques récupérés par *retrieval-augmented generation*,
- les Observations FHIR récentes (biosignaux).
   MedGemma 4B peut alors décider de demander un capteur précis et de générer un rapport structuré. ([Google for Developers](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card?utm_source=chatgpt.com))

## 3. Détails par couche

### 3.1 Application mobile

- **Stack** : Flutter + Dart (iOS/Android) ou Kotlin Multiplatform.
- **Fonctions clés** :
  - Authentification biométrique + OAuth PKCE.
  - Gestion du cache local chiffré (SQLCipher).
  - Wrapper BLE (GATT) conforme IEEE 11073-PHD pour autoscan et appairage capteurs. ([Wikipedia](https://en.wikipedia.org/wiki/ISO/IEEE_11073?utm_source=chatgpt.com))
  - Capture image → conversion DICOM ou PNG + métadonnées EXIF (anonymisées).
- **Sécurité** : stockage minimal de PHI, effacement à la demande (RGPD art. 17).

### 3.2 Dispositif biosignaux

- Microcontrôleur ARM Cortex-M + Zephyr RTOS, Bluetooth LE 5.3, profil **Health Device Profile**.
- Capteurs modulaires : ECG 1-12 dérivations, PPG (SpO₂), tensiomètre oscillométrique, thermistor, glucomètre optique.
- Firmware expose un *Device Information Service* décrivant ses capacités ; l’orchestreur traduit en `Device` + `CapabilityStatement` FHIR. ([FHIR Build](https://build.fhir.org/ig/HL7/phd/?utm_source=chatgpt.com))

### 3.3 API Gateway

- Kong ou Envoy, mutual TLS, *rate limiting* patient vs. clinicien.
- Contrôle d’accès basé rôle : Patient, Médecin, Super-admin.

### 3.4 Orchestrateur

- Python ( FastAPI ) + Celery.
- Compose un `PromptPackage` (JSON) : texte, URLs S3 signées des images, Observations, policy.
- Appelle l’inférence en streaming (gRPC).
- Convertit la réponse structurée JSON→FHIR.

### 3.5 Service d’inférence

- Modèle **medgemma-4b-it** en quantisation INT8, chargé par Triton ou Llama-cpp-server.
- Auto-scaling Kubernetes : GPU NodePool « burst ».
- Audit : enregistre *inputs*/*outputs* dans une base append-only (WORM) chiffrée.

### 3.6 Datastore clinique

- **FHIR R5** sur PostgreSQL (Google Cloud Healthcare API ou Firely Server).
- **Vector store** pour RAG : embeddings *text-med with-gemma* stockés dans pgvector.
- **Objet** : MinIO + SSE-KMS (AES-256).

### 3.7 Portail clinicien

- React + Next.js + FIDO2 (clé NFC) pour signature.
- Workflow BPMN : *In review → Approved → Issued* ; déclenche envoi d’e-prescription e-CPS / DMP.

## 4. Flux principaux

1. **Initialisation** : l’appli découvre les capteurs disponibles et publie le `Device` FHIR.
2. **Conversation** : les échanges sont streamés; la sortie *assistant* peut contenir l’instruction `<request_observation code="8867-4"/>` (ex. FC). L’appli déclenche la mesure, renvoie Observation.
3. **Rapport** : quand l’utilisateur choisit « Clôturer », l’orchestreur demande à MedGemma : `## Output = DiagnosticReport+MedicationRequest (JSON-FHIR)`. Le bundle reste en **Draft** jusqu’à validation.
4. **Validation** : le médecin reçoit une notification, révise, signe; une copie PDF/A du rapport est générée et stockée.
5. **Archivage** : tous les artefacts vers DMP / EHR via IHE MHD.

## 5. Conformité réglementaire (UE)

| Domaine                   | Référence                                                    | Application                                                  |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| QMS                       | **ISO 13485**                                                | Processus de dev. documenté                                  |
| Cycle logiciel            | **IEC 62304** SaMD                                           | Classification B/C selon risque clinique ([Regulatory knowledge for medical devices](https://blog.johner-institute.com/category/iec-62304-medical-software/?utm_source=chatgpt.com)) |
| Santé logicielle autonome | **IEC 82304-1**                                              | Preuve de sécurité & performance ([Medical Device HQ](https://medicaldevicehq.com/articles/the-illustrated-guide-to-software-as-a-medical-device-samd-iec-82304-1-and-ai/?utm_source=chatgpt.com)) |
| Risque                    | **ISO 14971**                                                | Analyse PHA / FMEA                                           |
| IA                        | MDCG 2024-1 + Plan d’Investigation Clinique pour IA adaptative |                                                              |
| Protection données        | **RGPD** + EDPB 04/2020                                      | DPIA, chiffrage, effacement, logs                            |
| Dispositif                | **MDR 2017/745** Classe IIa (diagnostic support)             |                                                              |
| Interop.                  | **HL7 FHIR** R5, **ISO/IEEE 11073**, **DICOMweb**            |                                                              |

Des *evidence packs* (tests unitaires, revues de code, rapports de vérification, journal d’audit) doivent accompagner le dossier technique CE.

## 6. Évolutivité et observabilité

- **CI/CD** : pipelines GitLab, analyse SAST/DAST, exécution d’essais de non-régression clinique.
- **Monitoring** : Prometheus + Grafana (latences, usage GPU, incidents), alertes sur SLI > 99 %.
- **Journalisation** : AuditEvent FHIR ; export immutable (WORM-S3).
- **Feature flags** : stratégies *canary* pour nouveaux modèles.

## 7. Feuille de route capteurs

| Phase | Capteurs                           | Norme de données          | Nouveaux prompts possibles                  |
| ----- | ---------------------------------- | ------------------------- | ------------------------------------------- |
| M-0   | ECG 1-dér., PPG, BP                | IEEE 11073, LOINC         | « Placez deux électrodes sur l’avant-bras » |
| M-1   | Spiromètre, Thermomètre tympanique | FHIR Observation + SNOMED | « Soufflez dans l’embout 3 s »              |
| M-2   | Glucomètre CGM, Oxymètre nocturne  | Continua BLE              | « Scannez votre capteur glycémie »          |

Le prompt *système* inclut la liste JSON des capteurs actifs ; MedGemma indique par `needs_data` les mesures nécessaires.

## 8. Résumé décisionnel

- **MVP** : hébergement cloud (Vertex AI) pour limiter DevOps GPU, mode Dialogue seulement.
- **Phase 2** : ajout Rapport structuré, portail clinicien, stockage FHIR.
- **Phase 3** : intégration capteurs BLE standardisés, extension vers edge-LLM on-device (éventuellement Gemma 2B quantisée) pour confidentialité partielle.

Une telle architecture sépare strictement interface patient, logique clinique, IA, et validation médicale, tout en respectant les exigences SaMD et RGPD. Elle offre l’évolutivité nécessaire pour intégrer de nouveaux biocapteurs et modèles plus puissants sans refonte complète.

### 1. Objectif et périmètre du **MVP**

Construire un service *chat multimodal texte + images/documents* autour de **MedGemma-4B** pour un usage de test sur :

- une machine de développement : **RTX 4090** (24 Go)
- un serveur de pré-production : **GeForce 1080 Ti** (11 Go)

Aucun biocapteur dans cette première itération ; seul le dialogue, l’intégration documentaire (RAG) et l’upload d’images sont couverts.

------

### 2. Macro-architecture de référence

```
┌───────────────────────────────┐
│        Client (Web / App)     │
│ Flutter | React | CLI         │
└──────────────┬────────────────┘
               │ HTTPS (REST)
┌──────────────▼────────────────┐
│     API & Orchestrateur       │  (FastAPI)
│  • Auth (OAuth PKCE)          │
│  • Gestion du chat            │
│  • RAG (pgvector)             │
│  • Upload fichiers → MinIO    │
└──────────────┬────────────────┘
               │ gRPC / HTTP
┌──────────────▼────────────────┐
│  Service d’inférence LLM      │
│  • MedGemma-4B INT4           │
│  • llama.cpp / vLLM           │
└──────────────┬────────────────┘
               │
┌──────────────▼────────────────┐
│      Stockages techniques     │
│  • PostgreSQL + pgvector      │
│  • MinIO objet (images/docs)  │
└───────────────────────────────┘
```

------

### 3. Détails des composants

| Couche                  | Technologies                                                 | Fonctions clés                                               |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **Client**              | Flutter (mobile) ou React / Next.js (desktop)                | **/chat** (stream SSE), **/upload** (multipart)              |
| **API + Orchestrateur** | Python 3.12, FastAPI, Uvicorn, SQLAlchemy                    | ① Auth PKCE ; ② historique conversation en BDD ; ③ pipeline RAG (chunk → embeddings → pgvector) ; ④ pré-prompt multimodal (texte + signed URLs) ; ⑤ streaming des tokens |
| **Service LLM**         | *Option A* : llama.cpp serveur, MedGemma-4B **INT4** (gpu-offload = 32) – ≤ 10 Go VRAM ; *Option B* : vLLM + Transformers (« load_in_4bit ») | Décodage streaming, API gRPC « Generate »                    |
| **BDD**                 | PostgreSQL 16, extension **pgvector**                        | stockage embeddings (cosine <20 ms sur 100 k chunks) ([DEV Community](https://dev.to/mattfergoda/ai-powered-image-search-with-clip-pgvector-and-fast-api-1f1d?utm_source=chatgpt.com)) |
| **Objets**              | MinIO (S3 compatible), bucket *chat-assets*                  | fichiers chiffrés SSE-KMS, URLs pré-signées 15 min           |
| **Infra**               | Docker Compose dev ; Helm + K3s quand RTX 4090 accessible à distance | séparation claire API / LLM pour swap GPU                    |

------

### 4. Contraintes GPU et chargement du modèle

- MedGemma-4B full-precision ≈ 14 Go VRAM ; en **int4** – ggml/QAT – VRAM ~ 8,5 Go : compatible 1080 Ti 11 Go. ([Medium](https://mychen76.medium.com/google-medgemma-an-open-model-that-excels-in-understanding-medical-text-and-images-b19526297d6a?utm_source=chatgpt.com))

- Commande (llama.cpp) :

  ```bash
  ./server -m medgemma-4b-it-q4.gguf \
           --gpu-layers 32 \
           --ctx-size 4096 \
           --mmproj models/medgemma_mmproj.gguf
  ```

- Sur RTX 4090 vous pouvez charger une version FP16 ou INT8 pour davantage de contexte (8 k-tokens).

------

### 5. Schéma des API

| Verbe  | URI                     | Payload                                    | Réponse                        |
| ------ | ----------------------- | ------------------------------------------ | ------------------------------ |
| `POST` | `/v1/chat`              | `{"session_id", "message", "files":[ids]}` | SSE tokens (`role`, `content`) |
| `POST` | `/v1/upload`            | multipart : fichier (png/jpg/pdf)          | `{file_id, presigned_url}`     |
| `GET`  | `/v1/files/{id}`        | –                                          | binaire signé                  |
| `GET`  | `/v1/history/{session}` | –                                          | JSON messages                  |

------

### 6. Pipeline RAG minimal

1. **Ingestion** :
    `ingest.py <file.pdf>` → pages ▸ sentence-chunks 512 tokens → embeddings CLIP-text (ou Gemma-text-embedding) → `pgvector`.
2. **À chaque requête** :
    – similarity search *k* = 6 → passages ;
    – contexte = system-prompt + passages + historique tronqué ;
    – images transmises via URL signée dans `image:<url>`.

------

### 7. Déploiement *docker-compose.yml*

```yaml
services:
  api:
    build: ./api
    environment:
      DATABASE_URL: postgres://...
      MINIO_ENDPOINT: http://minio:9000
      LLM_ENDPOINT: http://llm:8080
    depends_on: [db, minio, llm]
    ports: ["8000:8000"]

  llm:
    image: ghcr.io/ggerganov/llama.cpp:latest
    command: >
      ./server -m /models/medgemma-4b-it-q4.gguf
               --gpu-layers 32 --ctx-size 4096
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  db:
    image: postgres:16
    environment: { POSTGRES_PASSWORD: example }
    volumes: [dbdata:/var/lib/postgresql/data]
    command: >
      postgres -c shared_preload_libraries=pgvector

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minio
      MINIO_ROOT_PASSWORD: minio123
    volumes: [miniodata:/data]

volumes: { dbdata: {}, miniodata: {} }
```

------

### 8. Sécurité & conformité minimale

- **TLS** terminé par Traefik (Let’s Encrypt) ou Caddy.
- Auth **OAuth 2 PKCE** (Keycloak) ; jetons 30 min.
- Logs d’accès (**AuditEvent FHIR** ultérieurement) chiffrés, rotation locale (30 jours).

------

### 9. Roadmap de mise en œuvre

| Semaine | Livrable                                           | Points d’attention                  |
| ------- | -------------------------------------------------- | ----------------------------------- |
| **S-1** | Conversion modèle GGUF INT4, POC /curl sur 1080 Ti | vérifier consommation VRAM          |
| **S-2** | API FastAPI (`/chat`, streaming SSE)               | gestion timeout GPU                 |
| **S-3** | Upload image/doc, signature S3                     | taille max 20 Mo, antivirus ClamAV  |
| **S-4** | Pipeline RAG avec pgvector                         | quality : cosine ≥ 0.35             |
| **S-5** | Client mobile Flutter (chat + image picker)        | refresh OAuth silencieux            |
| **S-6** | Tests de charge (Locust) 5 req/s                   | file d’attente Celery si GPU saturé |

------

### 10. Évolutivité immédiate

- **Swap GPU** : placer `llm` sur une machine distincte (RTX 4090) ; exposer gRPC sur WireGuard ; aucune modification côté API.
- **Scaling CPU** : `api` stateless → répliques horizontalement.
- **Passage production** : basculer MinIO et Postgres vers services managés (RDS, Object storage).

------

## Conclusion

Cette architecture réduit au strict nécessaire : un backend Python unique pour l’orchestration, un service LLM isolé pour tirer parti de vos deux GPU, et deux stockages légers (pgvector + MinIO). Elle vous permet de tester rapidement le dialogue multimodal sur 1080 Ti, d’itérer la qualité du prompt et du RAG, puis de migrer sans rupture vers un GPU plus puissant ou vers le cloud.

### Arborescence minimale du **backend Python**

```
backend/
├── app/
│   ├── __init__.py
│   ├── config.py            # Paramètres centralisés (env, chemins, clés)
│   ├── main.py              # Point d’entrée FastAPI + middlewares
│   ├── dependencies.py      # Injections (session DB, clients MinIO, LLM)
│   ├── db/
│   │   ├── database.py      # Moteur SQLAlchemy + sessionmaker
│   │   ├── models.py        # ORM : User, ChatSession, Message, File
│   │   └── schemas.py       # Pydantic (request / response)
│   ├── llm/
│   │   ├── __init__.py
│   │   └── medgemma.py      # Chargement PyTorch + BitsAndBytes INT4, génération
│   ├── rag/
│   │   ├── embeddings.py    # Génération embeddings (Gemma-text ou SigLIP)
│   │   └── retriever.py     # Recherche pgvector (k-NN / HNSW)
│   ├── storage/
│   │   ├── __init__.py
│   │   └── minio_client.py  # Upload, URLs pré-signées, antivirus optionnel
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py          # Endpoint /v1/chat (SSE ou WebSocket)
│   │   └── upload.py        # Endpoint /v1/upload, /v1/files/{id}
│   └── services/
│       ├── chat_service.py  # Orchestration prompt ↔ LLM ↔ RAG
│       └── rag_service.py   # Pipeline ingestion + indexation
├── scripts/
│   ├── ingest.py            # CLI : découpe PDF/HTML, vectorisation, DB load
│   └── test_gpu.py          # Script sanity-check VRAM, débit tokens
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

------

### Rôle précis de chaque fichier

| Fichier             | Responsabilité                                               | Points clefs d’implémentation                                |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **config.py**       | Classe `Settings` (Pydantic BaseSettings) : URL DB, creds MinIO, chemin modèle, {MAX_TOKENS, TEMPERATURE}. | Lecture `.env`, validation, accès via `from app.config import settings`. |
| **main.py**         | Crée l’app FastAPI, monte les *routers*, CORS/TLS, Prometheus metrics, exception handlers. | Lance Uvicorn ou Gunicorn-Uvicorn workers.                   |
| **dependencies.py** | Fonctions `get_db()`, `get_storage()`, `get_llm()` injectées via `Depends`. | Permet de swapper le client LLM (localhost 1080 Ti ↔ remote 4090) sans toucher au code métier. |
| **database.py**     | Initialise `engine = create_engine(DB_URL, pool_pre_ping=True)` ; configure `SessionLocal`. | Active l’extension `pgvector` au démarrage.                  |
| **models.py**       | Tables : `ChatSession(id, user_id, created)`, `Message(id, session_id, role, content, created)`, `File(id, owner_id, path, mime, size)`. | Index `GIN` sur `Message.embedding` pour hybrid search.      |
| **schemas.py**      | Objets Pydantic : `ChatRequest`, `ChatChunk`, `UploadResponse`, etc. | Validation taille fichier, type MIME.                        |
| **medgemma.py**     | 1. Charge tokenizer & vision projector. 2. Configure **BitsAndBytesConfig** (`load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`) ([Hugging Face](https://huggingface.co/docs/transformers/main/model_doc/gemma?utm_source=chatgpt.com)). 3. Pipeline `generate_stream()` qui rend un générateur de tokens. | Consommation VRAM ≈ 8–9 Go sur 1080 Ti grâce à l’INT4 GQA ([PyTorch](https://pytorch.org/blog/int4-decoding/?utm_source=chatgpt.com)). |
| **embeddings.py**   | Instance SentenceTransformer ou `AutoModel.from_pretrained("google/gemma-text-embed")`; fonction `embed(text | image)`.                                                     |
| **retriever.py**    | Requête `SELECT ... ORDER BY embedding <=> :q LIMIT 6` (cosine) + index HNSW pour latence < 10 ms ([Google Cloud](https://cloud.google.com/blog/products/databases/faster-similarity-search-performance-with-pgvector-indexes?utm_source=chatgpt.com)). | Renvoie passages + score.                                    |
| **minio_client.py** | Génère URL PUT 15 min, URL GET 15 min, supprime fichiers orphelins. | Option ClamAV scan avant persistance.                        |
| **upload.py**       | Route multipart ; stocke fichier, retourne `{file_id, presigned_url}`. | Taille max paramétrable via settings.                        |
| **chat.py**         | Route `/v1/chat` : - insère message user, - récupère contexte RAG, - appelle `generate_stream()`, - boucle `await resp.write(event_json)`. | Support SSE par défaut, WebSocket possible.                  |
| **chat_service.py** | Assemble `system_prompt`, historique tronqué, passages RAG, balises `<image>URL</image>` ; gère stratégie de coupure `max_tokens-2 k`. | Implémente *retry* avec file d’attente si GPU occupé.        |
| **rag_service.py**  | `ingest(file_path)` → chunks, embeddings, `INSERT`.          | Nettoyage HTML, OCR optionnel (Tesseract).                   |
| **ingest.py**       | CLI asynchrone ; lit fichiers du dossier, appelle `rag_service.ingest`. | À lancer manuellement ou en tâche cron.                      |
| **test_gpu.py**     | Lance un prompt court, mesure temps moyen par token pour valider débit. | Affiche VRAM utilisée via `nvidia-smi`.                      |

------

### Flux principal `chat_service`

```text
User message
   ↓
Message enregistré en base
   ↓
retriever.similarity_search(query)              # RAG
   ↓
Prompt = SYSTEM + passages + history + <image>
   ↓
llm.generate_stream(prompt, images)
   ↓
Chunks SSE → client
   ↓
Chunks stockés, session persistée
```

------

### Dépendances (requirements.txt)

```
fastapi[all]==0.111.*
uvicorn[standard]==0.29.*
sqlalchemy==2.0.*
psycopg[binary]==3.1.*
pgvector==0.6.*
pydantic-settings==2.*
minio==7.*
torch>=2.2
transformers>=4.41
accelerate>=0.30
bitsandbytes==0.43
sentence-transformers>=2.7
python-multipart==0.0.*
aioboto3==12.*
```

------

### Bonnes pratiques de déploiement

- **Dockerfile** multi-stage : image base `python:3.12-slim`, puis `pip install --no-cache-dir -r requirements.txt`.
- Monter `/models` en volume pour changer le quantized GGUF ou Safetensors sans rebuild.
- Limiter à **1** worker Gunicorn si le GPU est local ; sinon placer l’API dans un conteneur CPU et le service LLM dédié GPU.
- Activer `trust_remote_code` **uniquement** sur l’hôte interne, jamais en production exposée.

------

#### Résultat

Avec cette structure, vous disposez :

- d’un **backend modulaire** où chaque couche (LLM, RAG, stockage) est isolée ;
- d’une **chaîne d’inférence PyTorch INT4** compatible 1080 Ti / 4090 ;
- d’APIs simples permettant de piloter la solution depuis un mobile, un navigateur ou toute application tierce.

### Ordre de rédaction conseillé des fichiers — approche incrémentale

| Étape                              | Objectif immédiat                              | Fichiers à créer/modifier                                    | Contenu minimal à livrer avant de passer à l’étape suivante  |
| ---------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **1. Initialisation du projet**    | Disposer d’un dépôt exécutable « Hello World » | - **requirements.txt**- **Dockerfile**- **docker-compose.yml** | • Listez seulement FastAPI, Uvicorn et Pydantic.• Dockerfile : image *python:3.12-slim*, installation des requirements, `CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]`.• Compose : service `api` exposant le port 8000. |
| **2. Squelette FastAPI**           | Lancer un endpoint `/health`                   | - **app/\**init\**.py**- **app/main.py**                     | `main.py` doit :1. Instancier `FastAPI()`.2. Définir `/health` qui retourne `{status:"ok"}`.3. Être exécutable via `uvicorn`. |
| **3. Configuration centralisée**   | Disposer de réglages chargeables par env vars  | - **app/config.py**                                          | Classe `Settings(BaseSettings)` avec : `DB_URL`, `MINIO_ENDPOINT`, `MODEL_PATH`, `MAX_TOKENS`, `TEMPERATURE`. Fournir `settings = Settings()` en module-level. |
| **4. Persistance SQL**             | Accès base Postgres sans tables                | - **app/db/database.py**- modifier **docker-compose.yml** (service `db`) | `database.py` :• `engine = create_engine(settings.DB_URL, pool_pre_ping=True)`.• `SessionLocal = sessionmaker(...)`. |
| **5. Modèles ORM et schémas**      | CRUD minimal pour le chat                      | - **app/db/models.py**- **app/db/schemas.py**                | Implémentez uniquement `ChatSession` et `Message` + schémas `ChatRequest`, `ChatChunk`. |
| **6. Dépendances d’injection**     | Accéder proprement à DB et settings            | - **app/dependencies.py**                                    | Fonctions `get_db()` (yield), `get_settings()` (return settings). |
| **7. Route \**/v1/chat\** (echo)** | Première route métier qui répond               | - **app/routers/\**init\**.py**- **app/routers/chat.py**     | Dans `chat.py`, enregistrez le router et renvoyez simplement `"echo": message`. Montez-le dans `main.py`. |
| **8. Service LLM (stub)**          | Séparer la couche d’inférence                  | - **app/llm/\**init\**.py**- **app/llm/medgemma.py**         | Dans `medgemma.py`, créez `class MedGemmaStub` avec méthode `generate_stream()` retournant un itérateur sur `"stub"`. |
| **9. Integration LLM ↔ API**       | Remplacer l’echo par le stub                   | - modifier **chat_service.py** (à créer)                     | `chat_service.chat()` appelle le stub et stream le résultat vers `chat.py`. |
| **10. RAG – structure vide**       | Préparer l’extension ultérieure                | - **app/rag/embeddings.py**- **app/rag/retriever.py**- **app/services/rag_service.py** | Placez uniquement des signatures de fonctions (`pass`) afin que les imports fonctionnent. |
| **11. Téléversment fichiers**      | Gestion MinIO basique                          | - **app/storage/\**init\**.py**- **app/storage/minio_client.py**- **app/routers/upload.py** | `minio_client.py` : instancier un `Minio` depuis `settings` et exposer `put_file`, `presign_get`.Route `/v1/upload` retourne un id et l’URL pré-signée GET. |
| **12. Implémentation LLM réelle**  | Passage au modèle quantisé 4-bit               | - enrichir **app/llm/medgemma.py**                           | • Charger tokenizer & modèle via `AutoModelForCausalLM.from_pretrained` avec `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`.• Supporter paramètre `images: List[str]` (chemins temporaires). |
| **13. Embeddings & recherche**     | Activer RAG textuel simple                     | - compléter **embeddings.py**, **retriever.py**, **rag_service.py** | • Utiliser `SentenceTransformer("all-MiniLM-L6-v2")`.• Insertion dans table `DocumentChunk(id, text, embedding)`. |
| **14. Prompt orchestration**       | Concaténer history + passages + balises image  | - finaliser **chat_service.py**                              | Ajoutez la logique de tronquage tokens et d’injection `<image>` + passages RAG. |
| **15. Scripts utilitaires**        | Outillage développeur                          | - **scripts/ingest.py**- **scripts/test_gpu.py**             | `ingest.py`: CLI `python ingest.py <path>` pour charger un PDF.`test_gpu.py`: envoie un prompt court, chronomètre le temps par token. |
| **16. Tests & documentation**      | Qualité de base                                | - **tests/** (pytest)                                        | Couvrir `/health`, `/v1/chat` stub, `/v1/upload`. Ajouter un *Makefile* cible `test`. |
| **17. Durcissement & métriques**   | Observabilité initiale                         | - enrichir **main.py**                                       | Middleware Prometheus, handler d’erreurs, CORS, rate-limit.  |

#### Règle d’or

Ne passez à l’étape *n+1* que lorsque les tests unitaires et `docker compose up` passent en vert pour l’étape *n*. Cette progression garantit un backend exécutable à chaque commit, tout en permettant d’ajouter les couches de complexité (LLM, RAG, MinIO) de façon contrôlée.

Ainsi vous écrivez d’abord le squelette minimal (config, main, health), puis les dépendances structurelles (DB, modèles, routage), avant d’implémenter progressivement l’inférence, le RAG et enfin le téléversement d’images.