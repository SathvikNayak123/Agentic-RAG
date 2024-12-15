# RAG Medical Assistant

Built a RAG Medical Assistant with Fine-tuned Llama-3.1-8b model.

![architecture](docs/1_lBVfMJ__9NjgKYiKI6mp4A.png)

## Features

- **Context-Aware Responses**: Provides precise medical advice by integrating over **20+ medical resources** through a RAG pipeline.
- **Efficient Document Retrieval**: Utilizes **LangChain** and **ChromaDB** for optimized and contextually accurate document retrieval.
- **Fine-Tuned LLaMA 3.1 Model**: Achieved superior performance with a **fine-tuned LLaMA 3.1 8B** model using **LoRA** techniques, achieving a **0.29 ROUGE score**.
- **Optimized Training**: Leveraged the **Unsloth library** for faster training and fine-tuning with **4-bit quantization**, significantly reducing resource usage without compromising performance.
- **Model Deployment**: Uploaded the optimized model to **Hugging Face** in **GGUF format**, enabling seamless integration and efficient inference.
- **Asynchronous Chat Interface**: Built with **FastAPI** to ensure low-latency and seamless user interaction, reducing response time by **40%**.

---

## Tech Stack

### 1. Language Model
- **LLaMA 3.1 (8B)** fine-tuned on **medical coversational datasets** using **PEFT (LoRA)** for domain-specific expertise.
- **Unsloth**: Used for efficient 4-bit quantization, reducing memory and computational costs during training and inference.
    ```bash
    https://GitHub.com/unslothai/unsloth.git
    ```
- **Ollama**: Used for model integration and serving.

### 2. RAG Pipeline
- **LangChain**: Enables integration of the LLaMA model with document retrieval capabilities.
- **ChromaDB**: Stores and retrieves embeddings for efficient and accurate context-aware responses.

### 3. Backend
- **FastAPI**: Provides a robust and asynchronous backend for a seamless chat interface.

### 4. Other Tools
- **Hugging Face**: Used for model hosting and inference, including support for GGUF model format.

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/SathvikNayak123/chatbot.git
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup**
   - Populate the database with medical documents.
   - Generate and store embeddings using the pre-trained LLaMA 3.1 model.
   - Install Ollama & pull model from HuggingFace
        ```bash
        ollama pull hf.co/sathvik123/llama3-ChatDoc
        ```

4. **Run the Application**
   ```bash
   uvicorn app:app --reload
   ```

---

## Result

- The Fine-tuned LLaMA3 model gave an **0.29 ROUGE score**

![sample-chat](docs/Screenshot%202024-12-15%20153236.png)


