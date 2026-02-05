# ToolLeak: Data Extraction Attacks on Knowledge-Based Tools in LLM Agents

ToolLeak is a research codebase for studying **data extraction attacks** against **knowledge-based tools** in LLM agents.  
This repository packages datasets as tools, supports two adversary settings, runs attacks, and reports evaluation metrics.

> ⚠️ Research Use Only  
> This project is intended for authorized security research and defensive evaluation. Do not use it to access or extract data from systems you do not own or do not have explicit permission to test.

---

## Environment Setup

### Requirements

- Python `3.9.0`

### Install Dependencies

Install dependencies from `requirement.txt`:

Run: 
```bash
pip install -r requirements.txt
```

### Optional: Use Ollama as the Agent LLM Backend

If you want to use **Ollama** as the core LLM for the agent, install Ollama first.

Linux installation command:

Run: 
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
---

## Datasets

This repository packages the datasets used in experiments **as tools**, stored under the `tools/` directory.

### Database-backed Datasets (Marketing & Pokemon)

Due to anonymization constraints, we do **not** provide the raw database files for the two database-based datasets.  
However, we provide scripts under the `database/` directory to build the experiment-ready databases.

#### Marketing Dataset

Create the database:

Run: 
```bash
python database/email_creat.py`
```

#### Pokemon Dataset

Create the database:

Run: 
```bash
python database/pokemon.py
```

---

## Preparation

### Adversary-1 Setup

Under the Adversary-1 setting, you must put **all tool names** and **high-level descriptions** into a single JSON file.

Important notes:

- ToolLeak does **not** depend on tool names semantically.
- Tool names are only used as **indices**.
- You can name tools arbitrarily (e.g., `"apple_and_pineapple"`).

A sample JSON file is provided under `tooldata/`.  
Use it as a template and fill in the tool list accordingly.

### Adversary-2 Setup

Under the Adversary-2 setting, you need to generate simulated tool functionality using the script in `attack/`.

Generate the simulated tools:

Run: 
```bash
python attack/TCL_black.py
```

---

## Run Attacks

### Adversary-1

Run: 
```bash
python main_adv1.py
```

### Adversary-2

Run: 
```bash
python main_adv2.py
```

---

## Evaluation

During attack execution, the framework automatically prints the following metrics:

- `EC`
- `ASR_C`
- `ASR_O`

To compute the additional metrics:

- `CRR`
- `SS`

Run: 
```bash
python metri.py
```

---

## Ethical Considerations

This work is intended to **identify and expose security vulnerabilities** in LLM agent systems for defensive research.

Any unauthorized use of ToolLeak to steal tool data or sensitive information from real-world deployments is strictly prohibited.  
Please conduct experiments only in lawful, authorized environments with explicit permission.
