# ClaimCheck


## How to run?

## Solution

I implemented and compared several solution strategies, designed across the three assignment phases. 


$${\color{red}Welcome} \space {\color{lightblue}To } \space {\color{lightgreen}Github}$$



### Phase 1: Zero-Shot Prompting (`ClaimCheck::ZeroFlat`)

In `ClaimCheck::ZeroFlat`, I designed a simple prompt that introduces the task and asks the model to classify a climate-related claim into one of the predefined sub-claim categories. The prompt includes an explanation of what a contrarian claim is, to help the model distinguish between non-claims and  contrarian arguments. Here is the prompt used in this phase:

```text
You are an expert in climate misinformation detection. Given a climate-related claim, your task is to classify it into one of the predefined claim categories listed below.

Contrarian claims are statements that reject, cast doubt on, or misrepresent the scientific consensus on human-caused climate change.

If the claim does not express a contrarian view, return 0: No Claim.

Choose only the most appropriate category and GENERATE ONLY THE NUMBER CORRESPONDING TO THAT CATEGORY. DO NOT GENERATE ANY OTHER WORD OR ANY EXPLANATION.
```

<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/ClaimCheck - Zero 2.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="500"/>


### Phase 2: Few-Shot Prompting (`ClaimCheck::FewShotFlat::[example_selection_strategy]`)

In the few-shot setting, I hypothesize that the type of examples included in the prompt has impact on model performance. To test this, I implemented several example selection strategies:
* One-per-Class: Includes exactly one example per sub-claim class, chosen to be as short as possible. (a.k.a., `ClaimCheck::FewShotFlat::OnePerClass`)
* Similarity-Based (RAG-like): Constructs a vector store over the training dataset and dynamically retrieves the most similar examples based on the input claim. (a.k.a., `ClaimCheck::FewShotFlat::Similar`)
* Counterfactual: Selects both the most similar and least similar examples for contrastive reasoning, helping the model better learn the decision boundaries. (a.k.a., `ClaimCheck::FewShotFlat::Counterfactual`)

<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/ClaimCheck - Few.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="500"/>

### Phase 3: Proposed Methods ("Whatever You Want")

As is mentioned in the assignment, advanced techniques such as full model fine-tuning and PEFT are valid paths. However, I deliberately chose not to use them for the following reasons:
* **Inflexibility for Experimentation**: Fine-tuning limits the solution to a specific model and specific prompt format. This reduces the flexibility to test multiple ideas such as dynamic example selection, hierarchical reasoning, or prompt-based logic, which were central to my approach.
* **Resource Limitations**: Even PEFT methods like LoRA/QLoRA require significant GPU memory and tuning, which is not ideal for lightweight environments like Google Colab. Prioritizing prompt-based methods aligns better with the assignment's compute assumptions.
* **Reasoning Over Memorization**: Based on my previous experience with LLMs, I hypothesize that modern LLMs are already strong in their reasoning capabilities. With carefully designed prompting techniques--such as Chain-of-Thought or retrieval-based example selection--it may be possible to match or even improve the performance of fine-tuned models. So, I prioritized prompt-based approaches to evaluate this hypothesis.


So, considering the above points, I propose the following methods:

#### Phase 3 - Solution 1: CoT Prompting
I explored two variants of CoT prompting:
  * Instructional CoT (`ClaimCheck::CoTFlat::Instruction`): The prompt explicitly outlines a step-by-step reasoning procedure and instructs the model to follow this structured thinking process when making predictions.
  * Few-shot CoT with Rationales (`ClaimCheck::CoTFlat::FewShot`): This variant adds example inputs to the prompt, each annotated with both a final label and a concise rationale explaining the reasoning behind the classification. These rationales were generated using an LLM and then curated and refined to ensure clarity and consistency across examples.

<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/ClaimCheck - CoT.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="500"/>


#### Phase 3 - Solution 2:  Hierarchical Classification (`ClaimCheck::ZeroHier`)

Since the classification task includes several classes, I designed a two-stage classification pipeline:
   * Claim Classification: Classify the type of the claim in the input.
   * Sub-Claim Classification: According to the detected claim class, classify the input into one of the relevant sub-claims,

This hierarchical structure can be applied on both zero-shot (`ClaimCheck::ZeroHier`) and few-shot (`ClaimCheck::FewShotHier::[example_selection_strategy]`) variants. Here is the general overview of this solution:
  
<img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/ClaimCheck - ZeroHier.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="1000"/>

#### Phase 3 - Solution 3: Contextual Bandit Learning (`ClaimCheck::Bandit`)
This approach combines a contextual bandit for **claim** classification with an LLM for **sub-claim** prediction.

* Why to use this appraoch instead of multi-arm bandit? I decided to use a contextual bandit instead of a multi-armed bandit because the optimal action depends on the input context.
* `ClaimCheck::Bandit` vs. `ClaimCheck::ZeroHier`: This approach uses lightweight bandit-based claim detection rather than using an llm as claim classifier to reduce computational cost. This is especially beneficial because the "0_0: No Claim" category appears frequently in the dataset; detecting it efficiently avoids unnecessary LLM calls.
* Task formulation as Contextual Bandit problem:
   * **State**: The [CLS] embedding of the input sentence, extracted by a BERT encoder.
   * **Context**: The semantic representation of the specific input sentence (state and context are equivalent here).
   * **Action**: The claim category predicted by the contextual bandit's MLP policy head.
   * **Reward**: Binary feedback (1 if the predicted claim matches the gold label, otherwise 0)

The below diagram shows the inference for this approach.
  <img src="https://github.com/ShirinTahmasebi/ClaimCheck/blob/main/ClaimCheck - Bandit.png" alt="ClaimCheck (Phase 1 and Phase 2)" width="1000"/>

# Evaluation

After carefully analyzing the dataset, I realized that a crucial challenge in this task is not just classifying the claims correctly, but also **identifying whether a sentence qualifies as a contrarian claim in the first place**. Many inputs contain declarative statements or proposals related to climate and policy, but they do not necessarily **reject, cast doubt on, or misrepresent** the scientific consensus on human-caused climate change.

For instance, the following example from the validation set is clearly a *claim*, but not a *contrarian* one. It advocates for a policy decision without challenging climate science itself. Without an explicit explanation of what is counted as a contrarian claim, the classifier is likely to mislabel such cases.


> *“Rather than raise our cost of energy to lower our emissions, it would make more sense to subsidize natural gas plants in developing countries to lower their reliance on coal and wood.”*




