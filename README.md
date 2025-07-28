# ClaimCheck


## How to run?

## Solution

### ClaimCheck0: Zero-shot Baseline

After carefully analyzing the dataset, I realized that a crucial challenge in this task is not just classifying the claims correctly, but also **identifying whether a sentence qualifies as a contrarian claim in the first place**. Many inputs contain declarative statements or proposals related to climate and policy, but they do not necessarily **reject, cast doubt on, or misrepresent** the scientific consensus on human-caused climate change.

For instance, the following example from the validation set is clearly a *claim*, but not a *contrarian* one. It advocates for a policy decision without challenging climate science itself. Without an explicit explanation of what is counted as a contrarian claim, the classifier is likely to mislabel such cases.


> *“Rather than raise our cost of energy to lower our emissions, it would make more sense to subsidize natural gas plants in developing countries to lower their reliance on coal and wood.”*


To account for this, I designed the zero-shot prompt to include a brief explanation of what defines a contrarian claim:

```text
You are an expert in climate misinformation detection. Given a climate-related claim, your task is to classify it into one of the predefined claim categories listed below.

Contrarian claims are statements that reject, cast doubt on, or misrepresent the scientific consensus on human-caused climate change.

If the claim does not express a contrarian view, return 0: No Claim.

Choose only the most appropriate category and GENERATE ONLY THE NUMBER CORRESPONDING TO THAT CATEGORY. DO NOT GENERATE ANY OTHER WORD OR ANY EXPLANATION.
```

