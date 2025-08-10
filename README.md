# ClaimCheck


In this small project, I propose and implement a set of solution named `ClaimCheck` set. Each of the solutions in `ClaimCheck` are described in detail in the [Solution](https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/README.md#solution) section. 

The codebase is designed to be super flexible: different types of LLMs, example selection strategies, or configuration settings can be swapped in or extended with minimal changes. 

To run any solution, simply execute the provided terminal command while specifying the solution name you wish to test. This modular design ensures that adding new methods or experimenting with alternative model configurations can be done quickly and cleanly.

```bash
python -m venv .claim_check_env
source .claim_check_env/bin/activate
pip install -r requirements.txt
```

Note: Put the HF Token in the `.env.development.template` and rename the file to `.env.development`

```bash
python flat_inference.py --solution "ClaimCheck::FewShotFlat::Similar"
```

Or, 

```bash
python hierarchical_inference.py --solution "ClaimCheck::ZeroHier"
```

The solution argument can be replace by the name of any other solution.

## Solution

I implemented and compared several solution strategies, designed across the three assignment phases. 

### Phase 1: Zero-Shot Prompting (`ClaimCheck::ZeroFlat`)

In `ClaimCheck::ZeroFlat`, I designed a simple prompt that introduces the task and asks the model to classify a climate-related claim into one of the predefined sub-claim categories. The prompt includes an explanation of what a contrarian claim is, to help the model distinguish between non-claims and  contrarian arguments:


<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/ClaimCheck - Zero 2.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="500"/>


### Phase 2: Few-Shot Prompting (`ClaimCheck::FewShotFlat::[example_selection_strategy]`)

In the few-shot setting, I hypothesize that the type of examples included in the prompt has impact on model performance. To test this, I implemented several example selection strategies:
* $${\color{purple}\text{One-per-Class}}$$: Includes exactly one example per sub-claim class. (a.k.a., `ClaimCheck::FewShotFlat::OnePerClass`)
* $${\color{purple}\text{Similarity-Based (RAG-like)}}$$: Constructs a vector store over the training dataset and dynamically retrieves the most similar examples based on the input claim. (a.k.a., `ClaimCheck::FewShotFlat::Similar`)
* $${\color{purple}\text{Counterfactual}}$$: Selects both the most similar and least similar examples for contrastive reasoning, helping the model better learn the decision boundaries. (a.k.a., `ClaimCheck::FewShotFlat::Counterfactual`)

<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/ClaimCheck - Few.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="500"/>

### Phase 3: Proposed Methods ("Whatever You Want")

As is mentioned in the assignment, advanced techniques such as full model fine-tuning and PEFT are valid paths. However, I deliberately chose not to priotorize them for the following reasons:
* **Resource Limitations**: Even PEFT methods like LoRA/QLoRA require significant GPU memory and tuning, which is not ideal for lightweight environments like Google Colab. Prioritizing prompt-based methods aligns better with the assignment's compute assumptions.
* **Reasoning Over Memorization**: Based on my previous experience with LLMs, I hypothesize that modern LLMs are already strong in their reasoning capabilities. With carefully designed prompting techniques--such as Chain-of-Thought or retrieval-based example selection--it may be possible to match or even improve the performance of fine-tuned models. So, I prioritized prompt-based approaches to evaluate this hypothesis.


So, considering the above points, I propose the following methods:

#### Phase 3 - Solution 1: CoT Prompting
I explored two variants of CoT prompting:
  * $${\color{teal}\text{Instructional CoT}}$$ (`ClaimCheck::CoTFlat::Instruction`): The prompt explicitly outlines a step-by-step reasoning procedure and instructs the model to follow this structured thinking process when making predictions.
  * $${\color{teal}\text{Few-shot CoT with Rationales}}$$ (`ClaimCheck::CoTFlat::FewShot`): This variant adds example inputs to the prompt, each annotated with both a final label and a concise rationale explaining the reasoning behind the classification. These rationales were generated using an LLM and then curated and refined to ensure clarity and consistency across examples.

<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/ClaimCheck - CoT.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="500"/>


#### Phase 3 - Solution 2:  Hierarchical Classification (`ClaimCheck::ZeroHier`)

Since the classification task includes several classes, I designed a two-stage classification pipeline:
   * $${\color{brown}\text{Claim Classification}}$$: Classify the type of the claim in the input.
   * $${\color{brown}\text{Sub-Claim Classification}}$$: According to the detected claim class, classify the input into one of the relevant sub-claims,

This hierarchical structure can be applied on both zero-shot (`ClaimCheck::ZeroHier`) and few-shot (`ClaimCheck::FewShotHier::[example_selection_strategy]`) variants. Here is the general overview of this solution:
  
<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/ClaimCheck - ZeroHier.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="1000"/>

#### Phase 3 - Solution 3: Contextual Bandit Learning (`ClaimCheck::Bandit`)
This approach combines a contextual bandit for **claim** classification with an LLM for **sub-claim** prediction.

* $${\color{gray}\text{Why to use this appraoch instead of multi-arm bandit?}}$$ I decided to use a contextual bandit instead of a multi-armed bandit because the optimal action depends on the input context.
*  $${\color{gray}\text{Comparing}}$$ `ClaimCheck::Bandit` $${\color{gray}\text{vs.}}$$ `ClaimCheck::ZeroHier`: This approach uses lightweight bandit-based claim detection rather than using an llm as claim classifier to reduce computational cost. This is especially beneficial because the "0_0: No Claim" category appears frequently in the dataset; detecting it efficiently avoids unnecessary LLM calls.
* Task formulation as Contextual Bandit problem:
   * **State**: The $${\color{olive}\text{[CLS]}}$$ embedding of the input sentence, extracted by a $${\color{olive}\text{BERT}}$$ encoder.
   * **Context**: The semantic representation of the specific input sentence (state and context are equivalent here).
   * **Action**: The claim category predicted by the contextual bandit's MLP policy head.
   * **Reward**: $${\color{olive}\text{Binary}}$$ feedback (1 if the predicted claim matches the gold label, otherwise 0)

The below diagram shows the inference for this approach.
  <img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/ClaimCheck - Bandit.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="1000"/>

#### Phase 3 - Solution 4: KL-Divergence Finetuning (`ClaimCheck::KL`)

In this approach, I leveraged an LLM with a classification head to predict a probability distribution over sub-claims. The key idea is that, instead of using only one-hot labels, we design a $${\color{brown}\text{hierarchical target distribution}}$$: the gold sub-claim receives the highest probability, other sub-claims from the same claim category receive a smaller share, and unrelated sub-claims receive minimal share. The model is trained to minimize the KL-divergence between its predicted distribution and this target, encouraging it to distinguish between claims and sub-claims.

This process is illustrated in the figure, showing the training phase, where KL-divergence loss guides parameter updates, and the inference phase, where the trained model outputs sub-claim probabilities for given inputs.  

<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/ClaimCheck - KL.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="1000"/>


# Evaluation



I implemented and evaluated several of the proposed approaches using the validation and test splits of the provided dataset. All experiments were conducted with the `google/gemma-2-2b-it` language model. Due to the limited computational resources available to me and the time required to run these experiments, I was unable to get the complete set of results. However, I would be happy to implement and execute the rest in the future if needed.



## Precision, Recall, F1 Score 

In this section,  I have three table to show the precision, recall, and F1 score accross the three methods , , and . The results are also reported per claim and per subclaim.

The figure below shows the results for `ClaimCheck::ZeroFlat`.

<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/ZeroFlat.png" alt="" width="500"/>

The results for `ClaimCheck::FewShotFlat::Similar` are also reported in the following figure. 

<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/FewShotFlat.png" alt="" width="500"/>

For `ClaimCheck::ZeroHier`, the results are as below: 

<p align="center"><img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/images/ZeroHier.png" alt="" width="500"/></p>

### Main Takeaways
Here are the main takeaways from the above three tables:
* Claim "0" / Subclaim "0_0" are trivial for all models.
* Except subclaim "0_0", `ClaimCheck::ZeroHier` underperforms compared to `ClaimCheck::ZeroFlat` and `ClaimCheck::FewShotFlat::Similar`, indicating that the hierarchical constraint may mislead when claims are more nuanced or overlapping.
* Performance is claim-dependent for  `ClaimCheck::ZeroFlat` and `ClaimCheck::FewShotFlat::Similar`. This means that certain categories benefit more from richer contextual examples, while others benefit from shorter, more focused prompts. Potential reasons for this might be:
  * Long Input Effect: Long few-shot prompts may confuse the model, especially for categories where training examples are long.
  * Position Bias: When the prompt is long, earlier examples can negatively affect predictions. However, class "0_0" is safe because its examples are short and less noisy.



## Confusion Matrix

To do more analysis, we need to have the confusion matrix to check whether errors are due to confusion between subclaims within the same claim vs. completely wrong claim prediction. The first row of images are for claim classification, and the second row for sub-claim classification.

<p align="center">
 <img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/src/results/images/confusion_matrix_claim_zero_flat.png" alt="ZeroFlat" width="30%"/>
 <img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/src/results/images/confusion_matrix_claim_few_similar_flat.png" alt="FewShotFlat" width="30%"/>
 <img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/src/results/images/confusion_matrix_claim_zero_hierarchy.png" alt="ZeroHier" width="30%"/>
</p>

<p align="center">
 <img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/src/results/images/confusion_matrix_subclaim_zero_flat.png" alt="ZeroFlat" width="30%"/>
 <img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/src/results/images/confusion_matrix_subclaim_few_similar_flat.png" alt="FewShotFlat" width="30%"/>
 <img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/src/results/images/confusion_matrix_subclaim_zero_hierarchy.png" alt="ZeroHier" width="30%"/>
</p>

### Main Takeaways
