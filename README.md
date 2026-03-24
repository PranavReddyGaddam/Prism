# Prism: LLM Explainability with Process Reward Models

A comprehensive explainability framework for large language models, demonstrating the application of Process Reward Models (PRM) to enhance mathematical reasoning transparency through step-by-step verification.

## Overview

This project implements an end-to-end system for training, deploying, and visualizing explainability features of language models fine-tuned with Process Reward Models. The system provides real-time insights into model reasoning through multiple visualization techniques including token confidence analysis, attention mechanisms, logit lens, and gradient attribution.

## Architecture

### System Components

**Backend (FastAPI)**
- RESTful API server providing generation and explainability endpoints
- Model orchestration layer supporting both local and remote model inference
- HTTP client for proxying requests to GPU-hosted models via ngrok tunnels

**Frontend (React + TypeScript)**
- Interactive dashboard for model interaction and visualization
- Real-time rendering of explainability metrics
- Responsive UI built with Tailwind CSS and shadcn/ui components

**Model Hosting (Google Colab)**
- GPU-accelerated inference for Gemma models
- Flask server exposing generation and explainability endpoints
- Public access via ngrok tunnel service

### Technology Stack

**Backend:**
- FastAPI (Python web framework)
- PyTorch + Transformers (model inference)
- httpx (async HTTP client)
- pyngrok (tunnel management)

**Frontend:**
- React 18 with TypeScript
- Vite (build tool and dev server)
- Recharts (statistical visualizations)
- D3.js (network graph visualizations)
- Tailwind CSS (styling framework)

**Infrastructure:**
- Google Colab (GPU compute)
- ngrok (secure tunneling)
- Git (version control)

## Features

### Model Capabilities

**Base Model**
- Google Gemma 3 4B parameter model
- Standard causal language modeling
- Direct answer generation

**Fine-tuned Model**
- LoRA adapter (rank=16, alpha=32) trained on PRM dataset
- Step-by-step mathematical reasoning
- Correctness verification labels ([CORRECT]/[WRONG])
- Enhanced reasoning transparency

### Explainability Features

**Token Confidence Analysis**
- Per-token probability scores across generated sequence
- Statistical aggregation (min, max, average)
- Area chart visualization with trend analysis

**Attention Visualization**
- Layer-specific attention weight matrices
- Multi-head attention pattern analysis
- Interactive heatmap with token-level detail

**Logit Lens**
- Layer-wise prediction evolution
- Intermediate representation analysis
- Probability distribution across model depth

**Gradient Attribution**
- Input token importance scoring
- Network graph visualization
- Explained behavior metrics

**Reasoning Steps Breakdown**
- Automatic parsing of step-by-step solutions
- Per-step confidence aggregation
- Quality indicators (high/medium/low confidence)

## Installation

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- Google Colab account (for GPU inference)
- ngrok account (for tunnel service)

### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your ngrok URL and Hugging Face token
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Google Colab Setup

1. Open a new Colab notebook with GPU runtime
2. Install required packages:
```python
!pip install flask pyngrok transformers torch peft
```

3. Load models and start Flask server (see `backend/COLAB_SETUP.md` for complete code)

4. Copy the ngrok URL and update `backend/.env`:
```
MODEL_BASE_URL=https://your-ngrok-url.ngrok-free.dev
```

## Usage

### Starting the Application

**Terminal 1 - Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

**Browser:**
Navigate to `http://localhost:5173`

### API Endpoints

**Generation:**
- `POST /generate/` - Single model text generation
- `POST /generate/compare` - Side-by-side model comparison

**Explainability:**
- `POST /explain/confidence` - Token confidence scores
- `POST /explain/attention` - Attention weight matrices
- `POST /explain/logit-lens` - Layer-wise predictions
- `POST /explain/attribution` - Gradient attribution
- `POST /explain/hidden-states` - Hidden state analysis

**Metadata:**
- `GET /models/` - List available models
- `GET /models/{model_id}` - Get model details
- `GET /health` - Server health check

### Example Request

```bash
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gemma-finetuned",
    "prompt": "What is 25% of 200?",
    "max_new_tokens": 256
  }'
```

## Performance

### Benchmarks

**Inference Performance:**
- Average inference time: 30.07s (Colab free tier)
- Logit lens computation: 1.09s
- Attribution analysis: 0.52s

**Accuracy Metrics:**
- Step format adherence: 90%+ target
- Answer correctness: Evaluated on diverse problem types
- Label accuracy: Verified against ground truth

### Performance Testing

Run automated benchmarks:
```bash
cd backend
./performance_benchmark.sh
```

Results saved to `backend/performance_results.txt`

## Project Structure

```
.
├── backend/
│   ├── main.py                          # FastAPI application entry point
│   ├── model_manager.py                 # Model orchestration and inference
│   ├── remote_model_client.py           # Colab HTTP client
│   ├── routers/
│   │   ├── generation.py                # Generation endpoints
│   │   ├── models.py                    # Model metadata endpoints
│   │   └── explainability_remote.py     # Explainability endpoints
│   ├── performance_benchmark.sh         # Performance testing script
│   ├── .env.example                     # Environment template
│   └── requirements.txt                 # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.tsx                      # Main application component
│   │   ├── components/
│   │   │   ├── kokonutui/
│   │   │   │   └── ai-prompt.tsx        # Prompt input component
│   │   │   └── visualizations/
│   │   │       ├── TokenFlow.tsx        # Token confidence chart
│   │   │       ├── AttributionGraph.tsx # Attribution network graph
│   │   │       ├── AttentionMatrix.tsx  # Attention heatmap
│   │   │       └── LogitLens.tsx        # Layer prediction viz
│   │   └── main.tsx                     # React entry point
│   ├── package.json                     # Node dependencies
│   ├── tsconfig.json                    # TypeScript config
│   ├── tailwind.config.js               # Tailwind config
│   └── vite.config.ts                   # Vite config
│
├── .gitignore                           # Git ignore rules
└── README.md                            # This file
```

## Development

### Adding New Models

1. Update `MODEL_CONFIGS` in `backend/model_manager.py`
2. Implement model-specific prompt formatting
3. Add model metadata for frontend display
4. Update Colab server if using remote inference

### Adding New Visualizations

1. Create component in `frontend/src/components/visualizations/`
2. Add corresponding API endpoint in backend
3. Integrate into `App.tsx` dashboard
4. Update state management for new data types

### Code Style

**Backend:**
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Document complex logic with inline comments

**Frontend:**
- Follow TypeScript best practices
- Use functional components with hooks
- Maintain consistent naming conventions

## Testing

### Manual Testing

Test individual endpoints:
```bash
# Health check
curl http://localhost:8000/health

# Generation
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{"model_id":"gemma-finetuned","prompt":"Test prompt","max_new_tokens":128}'
```

### Performance Testing

Run comprehensive benchmark suite:
```bash
cd backend
./performance_benchmark.sh
```

## Deployment Considerations

### Production Deployment

**Backend:**
- Use production ASGI server (Gunicorn + Uvicorn workers)
- Configure proper CORS origins
- Implement rate limiting
- Add request authentication
- Set up logging and monitoring

**Frontend:**
- Build production bundle: `npm run build`
- Serve static files via CDN
- Configure environment-specific API endpoints
- Enable compression and caching

**Models:**
- Consider dedicated GPU hosting (AWS, GCP, Azure)
- Implement model versioning
- Add model caching strategies
- Monitor GPU memory usage

### Security

- Never commit `.env` files to version control
- Rotate ngrok URLs regularly
- Validate all user inputs
- Implement request size limits
- Use HTTPS in production

## Troubleshooting

### Common Issues

**Backend not connecting to Colab:**
- Verify ngrok URL in `.env` is current
- Check Colab runtime is active
- Ensure Flask server is running in Colab
- Test ngrok URL directly in browser

**Frontend not receiving responses:**
- Confirm backend is running on port 8000
- Check browser console for CORS errors
- Verify API endpoint URLs in frontend code
- Inspect network tab for failed requests

**Model inference timeout:**
- Increase timeout values in `remote_model_client.py`
- Check Colab GPU availability
- Reduce `max_new_tokens` parameter
- Monitor Colab resource usage

**Visualization not rendering:**
- Check browser console for JavaScript errors
- Verify data format matches component expectations
- Ensure required dependencies are installed
- Clear browser cache and reload

## Contributing

This is an academic research project. For questions or collaboration inquiries, please contact the project maintainers.

## License

This project is developed for academic purposes. Model weights and datasets are subject to their respective licenses:
- Gemma models: Google's Gemma Terms of Use
- LoRA adapters: Project-specific license
- Third-party libraries: See individual package licenses

## Acknowledgments

- Google for Gemma model architecture
- Hugging Face for Transformers library
- FastAPI and React communities for excellent frameworks
- Google Colab for free GPU access

## Citation

If you use this work in your research, please cite:

```
@misc{prism2026,
  title={Prism: LLM Explainability with Process Reward Models},
  author={[Your Name]},
  year={2026},
  howpublished={\url{https://github.com/yourusername/prism}}
}
```

## Contact

For technical questions or collaboration opportunities, please open an issue on the GitHub repository.

---

**Version:** 1.0.0  
**Last Updated:** March 2026  
**Status:** Active Development
