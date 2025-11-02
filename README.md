# QwenLoraTimeSeries
The package contains a framework for training LLMs for Time Series Prediction. The underlying LLM is Qwen2.5 0.5B Instruct. 

## Installation

To install and use the package:

```git clone ```

```pip install -e .```

## Usage 
The model itself is a class. 
The model has two hyperparameters: learning rate and LoRA rank. 

For training, the data must be preprocessed using Qwen's built in 
tokenizer. This can be loaded using the function ```load model```,
found within ```qwen_lora/utility/processing```. 

Example scripts for data preprocessing, training, and evaluation can be found in the ```demo``` directory. 

## Report

A report evaluating the suitability of using LLMs for Time Series Analysis is found in ```report/main.pdf```. 

The plots in the report were generated using a Google Colab Notebook
using Google's A100 GPUs:

https://colab.research.google.com/drive/1XiTptHHlL3bJggGusTEjX80w0zfFpabo?usp=sharing

Within the report, all example predictions are found within the Appendix
for layout purposes. 

## Contact 
If any issues are found, please email at: pan31@cam.ac.uk

## Declaration of Use of Autogenerative Tools
I, Phong-Anh Nguyen Trinh declare that autogenerative tools were used in
creation of this package and report. The tools were used to debug code issues, improve plotting and 
Cursor.ai's autocomplete feature was also used for code completion.The tools were also used for Latex debugging and formatting. 









