# Projekt R 

## Project Overview

This repository is dedicated to the exploration of language model capabilities and the selection of an optimal configuration for a chatbot. The project's primary focus is on enhancing Large Language Models (LLMs) by incorporating new knowledge.

### Motivation

LLMs are proficient in understanding and generating text based on extensive training with large sets of textual data. Despite their impressive capabilities, there exists a limitation when it comes to responding to queries about documents on which LLMs have not been tested. This project addresses this gap by investigating methods to embed knowledge about specific documents.

### Key Objectives

1. **Knowledge Incorporation**: Explore techniques to integrate new knowledge into LLMs, enhancing their responsiveness to specific documents.

2. **Chatbot Development**: Develop a chatbot tailored for students in the "Introduction to Programming" course.

3. **LLM Model Selection**: Utilize a specific LLM model, Mistral-7B-OpenOrca, to generate contextually relevant responses.

4. **Document-Related Queries**: Enable the chatbot to respond to student queries related to three official documents of the "Introduction to Programming" course.

## Code Structure

The codebase is organized to facilitate understanding and collaboration:

- **/english_docs_od_mentora_txt**: Contains documents necessary for testing and training.
  
- **/final_test (branch: antonia)**: Records from the third testing phase with Mistral-7B-OpenOrca model configurations.

- **main.py**: The entry point for running the chatbot application.

## Getting Started

1. Clone the repository: `git clone [repository_url]`
  
2. Switch to the 'antonia' branch: `git checkout antonia`
  
3. Install dependencies: `pip install -r requirements.txt`
  
4. Run the chatbot application: `python main.py`

## Test Results Analysis

Explore the `/final_test` directory to review the performance results of the Mistral-7B-OpenOrca model configurations.


 
