# September 2026 LSTM translation experiments

Analysis date: 22 September 2026. English → French, greedy autoregressive translation.

The main result is that improving the recorded validation loss did not reliably improve the translations people actually read. Early continued training with reduced teacher forcing gave modest gains. Later, more aggressive reduction produced repeated fragments and poorer translations despite lower autoregressive validation loss. The trained ten-layer continuation did not fix this. The new six-layer, full-teacher-forcing model produces cleaner sentences than these late low-TF runs on many shared prompts, but still changes meanings, drops information, and occasionally repeats catastrophically. It does not consistently beat the best earlier checkpoint.

This report separates three sources of evidence: experiment observations recorded on 17 September, metadata and histories stored inside the checkpoints, and translations regenerated on 22 September. The [complete translation comparison](translations.md) and [raw results](results.json) accompany the examples below. The comparison is an experiment record, not a controlled test of teacher forcing as the only variable.

## 1. The working hypothesis from the experiments

Adding depth while retaining learned parameters yielded little benefit in the recorded experiments. The later reductions in teacher forcing did not produce a worthwhile improvement in autoregressive quality within the available training budget. Although validation ran autoregressively and its loss decreased, outputs increasingly resembled plausible but unfaithful text.

The resulting strategy was to first obtain strong prediction quality with full teacher forcing, then consider a small reduction once the model is already accurate. The proposed stopping signal was a large loss increase during that fine-tuning. A related working hypothesis is that increasing capacity and improving basic prediction quality may be more efficient than teaching a weak model to recover from its own mistakes.

The saved experiments support the narrower practical observation: the late low-TF continuations were disappointing, and loss alone was a poor checkpoint-selection signal. They do **not** establish that reducing TF can never help, that full TF is sufficient for high-quality translation, or that scaling is the only way to address exposure bias. There was an earlier improvement during TF decay, and architecture, tokenization, data processing, and training duration also changed. No matched training wall-time or energy measurements are available to establish “faster and cheaper” experimentally.

## 2. Which models were actually compared?

Teacher forcing (TF) is the proportion of decoder inputs drawn from the correct target sequence during training. Translation itself receives no target tokens, regardless of the training ratio.

The old six-layer family has **32,598,427 parameters**, hidden size 256, and embedding dimension 42. The expanded ten-layer model has **47,311,259 parameters**, hidden size 256, and embedding dimension 42. The new six-layer model has **47,735,004 parameters**, hidden size 300, and embedding dimension 84. The new file’s `100mil` name is not its parameter count. All three configurations use attention and dropout 0.1.

<!-- MODEL_TABLE -->

| Run | Layers / parameters | Saved steps | Training TF at snapshot | Validation policy |
|---|---:|---:|---:|---|
| Early 53k | 6 / 32.60M | 53,503 | 0.9680 | TF=0 (historical record) |
| Baseline 103k | 6 / 32.60M | 103,503 | 0.8656 | TF=0 (historical record) |
| Decay 147k (archive) | 6 / 32.60M | 147,006 | 0.5994 | TF=0 (historical record) |
| Decay 172k | 6 / 32.60M | 172,000 | 0.3331 | TF=0 (historical record) |
| Decay best, interval 24 | 6 / 32.60M | Not saved | 0.20 (interval-24 history) | TF=0 (historical record) |
| Decay 224k | 6 / 32.60M | 224,012 | 0.2000 | TF=0 (historical record) |
| “No-TF” file, 228k | 6 / 32.60M | 228,000 | 0.2000 | TF=0 (historical record) |
| Ten-layer continuation | 10 / 47.31M | 64,000 | 0.0000 | TF=0 (historical record) |
| Full TF, noisy-v1 | 6 / 47.74M | 52,753 | 1.0000 | TF=1 (explicit metadata) |

<!-- END_MODEL_TABLE -->

The [checkpoint inventory](checkpoint-inventory.json) records all 13 inspected files under `~/checkpoints` and the new resumable checkpoint, including inference-only and expansion snapshots. The evaluation records exact paths, state-dictionary hashes, metadata, and histories. The saved 147k archive and the current 172k repository checkpoint have the same experiment ancestry but different weights. Their scores must not be interchanged merely because a historical report used the current file’s basename.

Two names need special care:

- `model-32mil-param-no-tf.latest.pth` has endpoint TF 0 in its planned schedule, but its saved current ratio is **0.20**. Its schedule starts the new decay at interval 26; the completed history ends at interval 25. It is not evidence for an already completed six-layer TF=0 experiment.
- The eight- and ten-layer `*-layers-expanded.pth` files are expansion initialization artifacts. Their metadata records copying a six-layer checkpoint at step 214,012 and adding layers; they contain no subsequent training history. They are not treated as trained model competitors. The separately saved `model-expanded.latest.pth` is a trained ten-layer run, with 64,000 steps in its own counter. Its metadata does not preserve an explicit initialization path, so the exact connection to a particular expansion artifact is not independently established here.

The expansion script copies compatible existing tensors but randomly initializes additional layers. Keeping old parameters does not preserve the original network function after new recurrent layers are inserted. Any benefit requires successful further training; the operation is not a free increase in translation quality.

## 3. What changed in the actual translations?

The following are direct regenerated outputs, with spelling, repetition, and punctuation left intact. The selected examples illustrate particular failure modes; they are not a random human-rated sample. All 29 probes across all compared models are in the appendix, including the live translation prompts and clean counterparts for noisy inputs.

<!-- TRANSLATION_TABLES -->

**English: A small mistake can cause another mistake.**

| Model | Actual French output |
|---|---|
| Decay 147k (archive) | Une petite erreur peut provoquer une autre erreur. |
| Decay 224k | Une petite erreur peut causer une erreur. |
| Ten-layer continuation | Une petite erreur peut une une erreur erreur erreur. |
| Full TF, noisy-v1 | Une petite erreur peut causer une autre erreur. |

**English: One wrong word can change the whole sentence.**

| Model | Actual French output |
|---|---|
| Decay 147k (archive) | Un mauvaise mot peut changer la peine de |
| Decay 224k | Un mot de mauvpeut peut ger ger toute peine rara.. |
| Ten-layer continuation | Un mauvinmot peut peut ger ger ger la peine ph... |
| Full TF, noisy-v1 | Un mot malade peut changer la totalité de la peine. |

**English: The model sometimes repeats words when it makes a mistake.**

| Model | Actual French output |
|---|---|
| Decay 147k (archive) | Le modèle repéparfois parfois les mots quand il fait une erreur. |
| Decay 224k | Le modèle parfois parfois parfois parfois mots mots mots lorsqu erreur erreur erreur. |
| Ten-layer continuation | Le modèle parfois parfois parfois mots mots mots mots lorserreur erreur erreur.. |
| Full TF, noisy-v1 | Le modèle reprend parfois les mots quand il fait une erreur. |

**English: My name is John. I am commander of the northern army. I am a general and a loyal servant to the true king.**

| Model | Actual French output |
|---|---|
| Decay 147k (archive) | Mon nom est John John. Je suis commandant de l'armée nordique. Je suis un général et un loyal loyal à la véritable. |
| Decay 224k | Mon nom est John. Je suis commandant de l ’ mée mée norNord. je suis un général et un loyoyoyoyal à la vrai. |
| Ten-layer continuation | Je suis nom mon John John. je commancommande de de de de mée Nord et général et et et un oyoyoyloyoy. oy...... |
| Full TF, noisy-v1 | Mon nom est John, je suis commandant de l'armée nord, et je suis un général à part entière au midi. |

**English: I hav a golden aple**

| Model | Actual French output |
|---|---|
| Decay 147k (archive) | I haha un dord |
| Decay 224k | I hhhun |
| Ten-layer continuation | Je hhde un |
| Full TF, noisy-v1 | Je suis un jeune homme japonais |

**English: I know John, but John does not know me.**

| Model | Actual French output |
|---|---|
| Decay 147k (archive) | Je sais John John John, mais John ne sme pas. |
| Decay 224k | Je sais John John John John mais John ne sais pas. |
| Ten-layer continuation | Je sais John John John John ne ne spas pas pas.. |
| Full TF, noisy-v1 | Je sais John John, mais je ne me sais pas. |

**English: The price decreased from 50 euros to 30 euros.**

| Model | Actual French output |
|---|---|
| Decay 147k (archive) | Le prix a baissé de 50 euros à 30 euros. |
| Decay 224k | Le prix a diminué de de 50 50 euros 30 30 euros. |
| Ten-layer continuation | Le prix a diminué de 50 50 à à euros 30 30 euros |
| Full TF, noisy-v1 | Le prix a diminué de 50 euros à 30 euros. |


<!-- END_TRANSLATION_TABLES -->

### Cleaner French can still carry the wrong meaning

The full-TF model frequently has better sentence structure than the late reduced-TF runs. That is a useful practical improvement, but grammatical appearance is not enough:

- In “One wrong word can change the whole sentence,” *mot malade* means a sick word, and *peine* is the punishment-related sense of “sentence.” The output has recognizable French structure while making two wrong lexical choices.
- “Russia is the largest country in the world” becomes *le pays le plus important du monde*. “Most important” does not preserve the claim about size; several older checkpoints preserve it correctly.
- “Heavy AI usage” becomes *l’utilisation d’un usage grave*. The AI concept is lost, “heavy” is misinterpreted, and the phrase becomes redundant. Preserving *positif* or *négatif* at the end does not repair that loss.
- “If the model predicts one wrong token” substitutes *baisse* for “token.” This is a wrong concept, not merely a stylistic preference.

For a simple successful case, the full-TF model translates “A small mistake can cause another mistake” as *Une petite erreur peut causer une autre erreur.* It also preserves both amounts and the direction of change in the 50→30 and 30→50 euro probes. Those local successes do not imply reliable performance on other predicates or sentence structures.

### Longer inputs and information loss

The commander/king prompt tests whether several propositions survive: identity, command of the northern army, rank, loyalty, service, and the true king. The full-TF output begins plausibly but finishes with *un général à part entière au midi*, omitting the servant/king relationship and inventing an unrelated ending. Late low-TF models retain some relevant fragments but often repeat or break them. Neither behavior is a faithful translation.

On a shorter prompt, “The meeting starts at 9:30 tomorrow morning” becomes *La séance débute à 9 h 30.* The time survives but “tomorrow morning” disappears. “I am waiting for my friend” becomes *Je suis attendu pour mon ami*, changing the action and grammatical role. These examples distinguish omissions and meaning changes from mere surface awkwardness.

### Typos and unfamiliar inputs

The new checkpoint records source typo augmentation probability 0.23 using nlpaug, and its embedded tokenizers exactly match `artifacts/tokenizers/noisy-v1`. Nevertheless, “I hav a golden aple” produces *Je suis un jeune homme japonais*, unrelated to the intended apple. “I knpw jon” becomes *Je suis heureux de dire*. Noise augmentation has not made these examples reliable.

The appendix pairs these with “I have a golden apple,” “I know John,” and “I don't know where Mike is.” The clean apple input already yields *J'ai une pomme germanique* (“a Germanic apple”), while the clean Mike sentence is translated correctly. Thus some failures precede the spelling noise, whereas others appear with the altered spelling or case. These are diagnostic examples, not a measured typo-robustness rate. The fragment “sixseven AI startup” has an unclear intended meaning, so its output can illustrate information loss but should not be given a precise correctness score without clarifying the intended phrase.

### Repetition is not eliminated by either policy

Late reduced-TF outputs visibly repeat short words and pieces of words, including on the sentence describing repetition itself. Full TF improves many of those examples, but “Tom felt safe” still produces *Trop show show show…* until the generation limit. This is a counterexample to any claim that full TF has already solved free-running instability.

The new model completes “I know John, but John does not know me” with *Je sais John John, mais je ne me sais pas.* It repeats the name, chooses the wrong French verb for knowing a person, and changes the subject of the second clause. The intended relation is asymmetric; a suitable translation is *Je connais John, mais John ne me connaît pas.* The criterion is preserving who knows whom, not matching that reference word for word.

## 4. Scores as supporting evidence

Every run uses the same 300 source/reference pairs from the local CSV and the same 300 from Tatoeba, inherited from the September 14 reports. Decoding is greedy, batch size 16, maximum 100 generated tokens, CPU, four computation threads. Scoring uses SacreBLEU 2.6.0, case-sensitive BLEU with 13a tokenization and default chrF2. Exact metric signatures and all 600 outputs per model are in `results.json`.

<!-- SCORE_TABLE -->

| Run | CSV BLEU | CSV chrF | Tatoeba BLEU | Tatoeba chrF | Length-capped outputs (CSV / Tatoeba) |
|---|---:|---:|---:|---:|---:|
| Early 53k | 27.43 | 48.15 | 17.59 | 42.14 | 1 / 0 |
| Baseline 103k | 28.79 | 50.13 | 19.23 | 44.87 | 1 / 1 |
| Decay 147k (archive) | 30.27 | 50.97 | 21.37 | 46.19 | 1 / 0 |
| Decay 172k | 27.85 | 49.62 | 20.29 | 44.57 | 1 / 0 |
| Decay best, interval 24 | 24.30 | 46.64 | 13.77 | 39.93 | 1 / 0 |
| Decay 224k | 25.46 | 47.83 | 14.18 | 40.62 | 1 / 0 |
| “No-TF” file, 228k | 25.25 | 47.15 | 13.78 | 39.29 | 1 / 0 |
| Ten-layer continuation | 16.49 | 39.83 | 7.28 | 35.01 | 1 / 0 |
| Full TF, noisy-v1 | 29.56 | 49.88 | 17.92 | 43.82 | 1 / 1 |

<!-- END_SCORE_TABLE -->

The earlier 103k→147k comparison improved BLEU from 28.79 to 30.27 on the CSV and from 19.23 to 21.37 on Tatoeba. The [historical analysis](../../../artifacts/evaluations/2026-09-14-quick-tf-decay/report.md) reported exploratory paired-bootstrap evidence for those BLEU increases, with less conclusive chrF changes. Additional training and TF reduction happened together, so that comparison cannot identify which caused the gain.

Later checkpoints do not continue that trend. The saved minimum-validation-loss checkpoint at interval 24 is not the best translator on these samples. The trained ten-layer run performs substantially worse, despite its lower historical autoregressive validation loss. Thus the available evidence is stronger than “no obvious improvement” for those specific late checkpoints: there is a measured regression on the shared samples, accompanied by visibly degraded outputs.

The full-TF model is better than the late reduced-TF and expanded runs on both samples, but below the 147k checkpoint on both BLEU and chrF. It is not uniformly better than the 103k baseline either. That is compatible with cleaner selected sentences and fewer broken fragments: readability and source fidelity are different dimensions, and model rankings depend on which examples and properties are examined.

No new significance claim is made for these nine-way comparisons. The two corpora may overlap. Some CSV references are visibly wrong: “The Commission shall comprise nine members” is paired with an unrelated presidency/rotation reference, while *La Commission comprend neuf membres* is a correct output. Single references also penalize valid alternatives. These problems motivate reading outputs, but they do not explain away failures such as replacing an apple with a Japanese man.

## 5. Reading the training graphs correctly

Open the [offline interactive training dashboard](assets/training-dashboard.html). It reproduces the notebook’s three views—train/validation loss, validation minus training loss, and TF schedule—from the saved checkpoint histories. It requires no external Plotly server. It is a new export from the same history fields, not a screenshot of a previously saved notebook output; the checked-in notebook has no saved plot outputs.

| Training history | Train loss, first → last | Validation loss, first → last | Training TF, first → last |
|---|---:|---:|---:|
| Old six-layer family, 25 logged intervals | 3.7548 → 4.0694 | 9.4298 → 5.4048 | 0.9940 → 0.20 |
| Trained ten-layer continuation, 7 intervals | 4.3942 → 4.7608 | 5.3634 → 4.7013 | 0.20 → 0.00 |
| New full-TF run, 11 intervals | 4.6199 → 1.8258 | 3.2517 → 1.7239 | 1.00 → 1.00 |

The old six-layer validation minimum is 5.2847 at interval 24, followed by 5.4048 at interval 25. Its training loss rises as the model receives fewer correct preceding tokens. This is a harder conditioning regime, so rising training loss is not automatically evidence of ordinary overfitting. Meanwhile, lower autoregressive validation loss did not guarantee better complete translations.

For the ten-layer run, the final train and validation losses are close once both use autoregressive conditioning. A small train/validation difference alone is not evidence of good translation: both can remain poor under the same difficult objective.

The full-TF model improves steadily under its own fixed policy. Its last validation loss of 1.7239 is encouraging for predicting target tokens when correct prefixes are available. It must not be ranked numerically against the older losses of 4.7–9.7: conditioning policy, tokenization, and loss aggregation differ. The new metadata explicitly records non-padding-token averaging and validation using the training TF ratio. The old artifacts omit these policy fields; zero-TF validation and historical batch averaging are documented by the earlier report and project history, consistent with the recorded experiment observations.

Training loss is accumulated while weights change and dropout is active; validation evaluates saved interval weights with dropout disabled. The new run also records source augmentation. A validation loss below training loss is therefore not itself suspicious. Plot horizontal axes are logged validation intervals, not equal amounts of compute: old runs allow up to 10,000 steps per interval, the new run 5,000, and dataset-pass boundaries can shorten intervals. The expanded model’s counter starts a new run rather than including the parent’s training.

## 6. Why loss and translation can disagree

The local implementation makes the distinction concrete. During full TF, the decoder gets the correct preceding target tokens. During partial TF, it sometimes gets its own argmax output instead. In free-running generation it always feeds back its own prediction and stops at EOS or the length cap.

Validation loss is still token cross-entropy against a fixed reference sequence. Under autoregressive conditioning, an earlier omission, extra word, or alternative phrasing can shift the generated prefix away from the reference. The remaining per-position prediction problem then differs from judging whether a complete French sentence faithfully expresses the English. A falling scalar loss can coexist with poorer word choices, repetition, or omissions. The present checkpoints demonstrate that coexistence; they do not identify a single cause for every error.

Likewise, full TF does not directly evaluate behavior after the model’s own mistakes. A low teacher-forced loss cannot guarantee stable generation. The observed repetition and drift are compatible with this training/inference difference, but tokenization, lexical coverage, optimization, model capacity, and data quality remain alternative or interacting explanations. All errors should not be attributed to exposure bias without a controlled ablation.

Validation itself does not update model weights. It influences monitoring, early stopping, and which checkpoint is selected; the training objective and schedule determine the gradient updates. Therefore, the observation is that the old validation criterion was insufficient for selecting useful translations, not that merely calculating autoregressive validation loss directly damaged the model.

## 7. Architectural constraints and the cost of autoregressive training

These costs follow from the current [model implementation](../../../src/lstm_translator/model.py) and [training loop](../../../src/lstm_translator/training.py). They explain the available optimization paths; the experiment histories do not contain a controlled timing comparison of training at different TF ratios.

### Recurrent depth and attention limit parallel work

The encoder is bidirectional, but the decoder generates left to right. Each decoder step uses the previous hidden state to compute attention over the encoder outputs, then passes the resulting context into its LSTM. The next step depends on that newly computed state. This attention-to-LSTM feedback keeps the decoder loop sequential **even with full teacher forcing**. Knowing all target input tokens does not remove that dependency in this architecture.

Depth adds work inside every step: each stacked layer depends on the layer below. The configuration’s hidden size is per encoder direction; the decoder concatenates those directions and has twice that width. Thus the new model’s hidden size of 300 means a decoder width of 600, not 300. Increasing recurrent width raises the cost of hidden-to-hidden matrix multiplications approximately quadratically; increasing depth repeats recurrent work across more layers.

Attention avoids forcing all source information through a single final vector, but each target step still scores the source positions. For source length S and target length T, that entails roughly S × T attention interactions per example, in addition to recurrent computation. The code caches the encoder projection once per forward pass, but must recompute the query-dependent scores and context at each target step. Longer BPE sequences therefore increase both decoding steps and attention work. None of these mechanisms guarantees correct alignment, preservation of names, or coverage of every source clause; the observed omissions cannot be assigned to one architectural cause from these runs alone.

Appending randomly initialized layers also changes the path taken by the pretrained representations. This implementation has no residual bypass that makes those new layers an identity mapping. Copying the existing tensors preserves their values, but the expanded network must learn to use them through the new layers. The disappointing ten-layer result is evidence about this particular expansion and continuation, not proof that deeper LSTMs always perform worse.

### Predicted-token feedback adds a costly dependency

At TF=1, all decoder input tokens are known. The current implementation embeds the target inputs together, runs the sequential attention/LSTM loop, and applies the vocabulary projection to the collected features in one operation. This batches substantial work even though recurrence itself remains sequential.

At TF<1, the code projects to the complete target vocabulary at **every step**, takes an argmax, and uses that predicted token for some or all examples in the next step. It cannot defer those vocabulary projections until the sequence is finished. Intermediate TF ratios also generate a per-example random mask and choose between predicted and reference tokens. Even a small reduction below 1 enters this per-step implementation path; its overhead need not scale linearly with the fraction of predicted tokens.

The main distinction is not that full TF avoids computing vocabulary logits—it computes them too. Full TF groups the projection work into a larger operation, whereas predicted-token feedback places repeated projections and token selection on the sequential dependency path. On accelerators, many small dependent operations can incur launch overhead and use hardware less efficiently than larger batched operations. The actual penalty depends on batch size, sequence lengths, vocabulary size, device, and backend; no speedup factor is established here.

### Training also retains the sequence for backpropagation

Autoregressive training is more expensive than simply generating a translation because it retains the differentiable recurrent computation and then backpropagates through it. The loop accumulates logits for every target position before computing cross-entropy and calling `backward()`. Logits occupy approximately B × T × V values for batch size B, target length T, and vocabulary size V, alongside recurrent and attention activations. Both full-TF and reduced-TF training pay these memory costs; using predictions does not turn training into memory-light inference.

The argmax token choice itself is discrete and does not carry a gradient. Gradients still flow through recurrent hidden-state connections and the per-position losses, but the model is not directly differentiating through the choice of an earlier wrong token to learn an explicit correction policy. Merely exposing it to more of its own errors therefore does not guarantee efficient error recovery.

The forward training loop runs to the padded reference length rather than stopping when a prediction emits EOS. Padding is excluded from the loss, but decoder computation still runs through those batch positions. Larger batches can improve throughput while increasing activation memory and padding waste. More layers, larger vocabularies, and longer sequences can constrain batch size further. Autoregressive validation adds another sequential forward pass, although it disables gradients and avoids backward-pass storage.

These constraints explain why reducing TF can consume substantial compute without a proportional gain in translation quality. Full TF has a more efficient path in the current implementation, but remains recurrent and costly. The practical questions are both how much each update costs and how much faithful translation improves per unit of training time; the latter needs matched training-time measurements, not just loss curves or inference timings.

## 8. Practical conclusion and the next experiment

For this project, the evidence favors improving basic translation fidelity under a stable full-TF training objective before spending another large budget on aggressive decay. That is a reasonable next experiment, not a claim that the current full-TF model has already reached high quality.

A useful continuation would retain a fixed, verified source-disjoint evaluation set and a small diagnostic suite for negation, numbers, names, word senses, multi-clause relations, and clean/noisy input pairs. Select checkpoints using generated outputs alongside teacher-forced loss. Track semantic omissions and repetitions explicitly; a low validation score alone failed to reveal these issues in the previous experiments.

To test a later small TF reduction, branch from the **same** strong checkpoint into a constant-TF continuation and a mildly reduced-TF continuation. Keep architecture, tokenizer, data ordering, validation policy, and budget comparable. Record actual wall time and processed tokens as well as steps. Treat increased loss as a signal to investigate, while checking generated quality; changing TF changes the loss task, so an increase alone is not a complete stopping rule. There is no validated optimal reduction magnitude in the current experiments.

Adding depth, widening the network, changing BPE, and adding typo augmentation should also be tested separately if the aim is causal understanding. The new run changes several of these factors at once. The present results cannot demonstrate that scale alone corrects exposure bias or that noisy-v1 is better or worse than the old tokenizer in isolation.

## 9. Reproduction and artifacts

From the repository root, with the project and notebook dependencies installed:

```bash
.venv-3-12/bin/python docs/experiments/2026-09-lstm/run_comparison.py
.venv-3-12/bin/python docs/experiments/2026-09-lstm/build_figures.py
.venv-3-12/bin/python docs/experiments/2026-09-lstm/build_tables.py
```

The checkpoint files are local, large, and not included in the documentation. The runner lists their expected paths under `~/checkpoints`, `~/new_model/checkpoints`, and the repository’s `artifacts/checkpoints`. Adjust those paths when reproducing elsewhere. `--models LABEL ...` reruns only selected models and retains the other saved entries. The figure builder reads the saved lightweight inventory; regenerating comparisons does not silently replace the historical inventory.

- [Detailed translation appendix](translations.md): all 29 diagnostic inputs and each model’s actual output.
- [Full evaluation results](results.json): corpus scores, references, predictions, diagnostic outputs, metadata, and model-weight fingerprints.
- [Checkpoint inventory](checkpoint-inventory.json): extracted architecture, training settings, schedules, and histories for all inspected checkpoints.
- [Training dashboard](assets/training-dashboard.html): independent plot axes for each run and policy.
- [Comparison runner](run_comparison.py), [figure exporter](build_figures.py), and [table builder](build_tables.py).

The evaluation reuses local historical examples; their absence from all training runs has not been verified. Tokenizer training exposure is also distinct from translation-model test separation. The manual probes are deliberately diagnostic and include failures noticed during use, so their frequency of errors cannot be interpreted as a population accuracy rate. The data, saved histories, and exact translations are sufficient to document these experiment outcomes, but not to claim production-level quality or a general result about all LSTM translation systems.
