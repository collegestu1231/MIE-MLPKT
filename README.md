# MIE-MLPKT
## MIE Framework
![MIE](MIE.png)
## MLPKT Framework
![MLPKT](MLPKT.png)

This repository is built for the paper MIE+MLPKT: Leveraging LLM and Multiscale Knowledge States to Improve Knowledge Tracing in Programming Tasks。
> **Author: mingxing Shao**, **tiancheng Zhang**
## Overview
This paper highlights three unique characteristics of programming knowledge tracing tasks compared to traditional subject-based knowledge tracing, which often lead to suboptimal performance of existing KT models on programming benchmarks. To address these challenges, we propose the MIE+MLPKT framework. The MIE component uses carefully designed prompts to extract reasonable scoring labels and multi-level error cause analyses from large language models (LLMs), mitigating label continuity and irrationality issues while reducing high uncertainty. To handle the multi-layered nature of student proficiency in programming tasks, we introduce MLPKT, a three-tiered knowledge tracing framework that integrates error analysis vectors from MIE. Extensive experiments across three datasets and 18 baselines validate the effectiveness of our framework. In future work, we aim to evaluate MIE+MLPKT in real-world educational settings.
## Reproduce
If you want to run the code, you should first download the "data" profile, available at: https://drive.google.com/drive/folders/1jdoNIi4GNeYeps6R3tCDsz3GAaEP7KK6?usp=drive_link. It is worth noting that since the paper has not yet been accepted, we have only open-sourced the first 5000 ratings and reason extraction data for each dataset. Once the paper is successfully accepted, we will immediately open-source the complete data.
