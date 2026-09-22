# Actual translations: complete diagnostic comparison

[Return to the analysis](README.md). Generated on 22 September 2026 with each checkpoint’s embedded tokenizer and vocabulary. The full-TF tokenizer state matches noisy-v1. Outputs are unedited; markup characters are escaped only for Markdown rendering. Maximum generated length: 100 tokens. These are purposive probes, not a representative accuracy sample.

The first 15 prompts reproduce the recorded live examples; the new model’s outputs match that transcript. Prompt 16 completes the input for which the transcript did not include an output. The remaining prompts add clean/noisy counterparts and tests of factual details. Checkpoint identities and model hashes are in [results.json](results.json).

## 1. Hello

| Model | Actual output |
|---|---|
| Early 53k | Hello |
| Baseline 103k | Hello |
| Decay 147k (archive) | Helloo |
| Decay 172k | H |
| Decay best, interval 24 | Hello |
| Decay 224k | Hello |
| “No-TF” file, 228k | Hell |
| Ten-layer continuation | Hello |
| Full TF, noisy-v1 | Hello |

## 2. How are you?

| Model | Actual output |
|---|---|
| Early 53k | Comment vous êtes? |
| Baseline 103k | Comment vous? |
| Decay 147k (archive) | Comment vous ez-vous? |
| Decay 172k | Comment vous ez-vous? |
| Decay best, interval 24 | Comment vous ez-vous? |
| Decay 224k | Comment vous ez-vous? |
| “No-TF” file, 228k | Comment vous ez-vous? |
| Ten-layer continuation | Comment vous ez-vous ez-vous? |
| Full TF, noisy-v1 | Comment vous êtes? |

## 3. Do u know how are you?

| Model | Actual output |
|---|---|
| Early 53k | Est-ce que vous avez besoin de quoi vous êtes? |
| Baseline 103k | Est-ce que tu tu sais comment tu tu? |
| Decay 147k (archive) | Vu savez comment vous vous? |
| Decay 172k | Au savez comment vous? |
| Decay best, interval 24 | Vu u savez comment vous? |
| Decay 224k | Estu savez comment vous vous?? |
| “No-TF” file, 228k | Estu savez comment vous? |
| Ten-layer continuation | Au -comment comment vous comment vous? |
| Full TF, noisy-v1 | Est-ce que vous savez comment vous êtes? |

## 4. Russia is the largest country in the world.

| Model | Actual output |
|---|---|
| Early 53k | La Russie est le plus grand pays du monde. |
| Baseline 103k | La Russie est le plus grand pays du monde. |
| Decay 147k (archive) | La Russie est le plus grand pays du monde. |
| Decay 172k | La Russie est le plus grand pays du monde. |
| Decay best, interval 24 | La Russie est le plus grand pays du monde. |
| Decay 224k | La Russie est le plus grand pays du monde. |
| “No-TF” file, 228k | La Russie est le plus grand pays du monde. |
| Ten-layer continuation | La Russie est le plus grand pays pays monde.. |
| Full TF, noisy-v1 | La Russie est le pays le plus important du monde. |

## 5. A small mistake can cause another mistake.

| Model | Actual output |
|---|---|
| Early 53k | Une petite erreur peut causer une autre erreur. |
| Baseline 103k | Une erreur erreur peut entraîner une erreur. |
| Decay 147k (archive) | Une petite erreur peut provoquer une autre erreur. |
| Decay 172k | Une petite erreur peut causer une autre erreur. |
| Decay best, interval 24 | Une petite erreur peut une une une erreur autre erreur. |
| Decay 224k | Une petite erreur peut causer une erreur. |
| “No-TF” file, 228k | Une petite erreur peut provoquer une erreur erreur. |
| Ten-layer continuation | Une petite erreur peut une une erreur erreur erreur. |
| Full TF, noisy-v1 | Une petite erreur peut causer une autre erreur. |

## 6. One wrong word can change the whole sentence.

| Model | Actual output |
|---|---|
| Early 53k | Un mot de mauvais traitements peut changer toute la phrase. |
| Baseline 103k | Un mot inconnu peut changer toute la phrase. |
| Decay 147k (archive) | Un mauvaise mot peut changer la peine de |
| Decay 172k | Un mauvais mot peut peut ger la phrase. |
| Decay best, interval 24 | Un mauvais mot peut peut ger ger toute la phrase. |
| Decay 224k | Un mot de mauvpeut peut ger ger toute peine rara.. |
| “No-TF” file, 228k | Un mauvais mots peut changer ger la phrase. |
| Ten-layer continuation | Un mauvinmot peut peut ger ger ger la peine ph... |
| Full TF, noisy-v1 | Un mot malade peut changer la totalité de la peine. |

## 7. I think the socio-economic impact of heavy AI usage may become negative rather than positive.

| Model | Actual output |
|---|---|
| Early 53k | Je crois que l'impact socio-économique de l'utilisation d'AI peut devenir négatif plutôt que positif. |
| Baseline 103k | Je crois que l ’ impact socioéconomique de l ’ utilisation de l ’ IAS peut devenir négatif plutôt que positif. |
| Decay 147k (archive) | Je crois que l ’ impact socio-économique de l ’ utilisation des l ’ II peut devenir négatif plutôt que positif. |
| Decay 172k | Je pense que l impact socio-économique de l'utilisation ’ de I peut devenir négatif plutôt que positif. |
| Decay best, interval 24 | Je pense que l'impact socio-économique de l'utilisation ’ de de peut peut devenir négatif négatif négatif. |
| Decay 224k | Je crois que l'impact socio-économique de l'utilisation ’ utilisation de I peut devenir devenir négatif plutôt que positif. |
| “No-TF” file, 228k | Je pense que l'impact socio-économique de l'utilisation lourde de l'AI pourrait devenir négatif négaplutôt que néga.. |
| Ten-layer continuation | Je crois que l'impact socio-économique économique de l'utilisation de de lourAI de néganéganéganéga. néganéga... |
| Full TF, noisy-v1 | Je crois que l’impact socio-économique de l’utilisation d’un usage grave peut être négatif plutôt que positif. |

## 8. I think the socio-economic impact of heavy AI usage may be positive.

| Model | Actual output |
|---|---|
| Early 53k | Je crois que l'impact socio-économique de l'utilisation d'AI peut être positif. |
| Baseline 103k | Je crois que l ’ impact socioéconomique de l ’ utilisation de l ’ IAS peut être positif. |
| Decay 147k (archive) | Je crois que l ’ impact socio-économique de l ’ utilisation de l ’ II peut être positif. |
| Decay 172k | Je pense que l impact socio-économique de l'utilisation ’ de I peut être positif. |
| Decay best, interval 24 | Je pense que l impact socio-économique de l'utilisation ’ utilisation de l ’ Ipeut peut positif positif. |
| Decay 224k | Je crois que l'impact socio-économique de l'utilisation ’ utilisation de I peut être positif positif. |
| “No-TF” file, 228k | Je pense que l'impact socio-économique de l'utilisation lourde de l'AI peut être positif positif. |
| Ten-layer continuation | Je crois que l'impact socio-économique économique de de utilisation de lourde de être.... |
| Full TF, noisy-v1 | Je crois que l’impact socio-économique de l’utilisation d’un usage grave peut être positif. |

## 9. The model sometimes repeats words when it makes a mistake.

| Model | Actual output |
|---|---|
| Early 53k | Le modèle répété parfois les mots lorsqu ’ il fait une erreur. |
| Baseline 103k | Le modèle répéparfois parfois les mots quand il fait une erreur. |
| Decay 147k (archive) | Le modèle repéparfois parfois les mots quand il fait une erreur. |
| Decay 172k | Le modèle reparfois parfois parfois parfois mots mots erreur. |
| Decay best, interval 24 | Le modèle parfois parfois parfois parfois les mots lorsqu ’ erreur erreur erreur. |
| Decay 224k | Le modèle parfois parfois parfois parfois mots mots mots lorsqu erreur erreur erreur. |
| “No-TF” file, 228k | Le modèle réparfois parfois parfois mots mots lorsqu ’ erreur erreur. |
| Ten-layer continuation | Le modèle parfois parfois parfois mots mots mots mots lorserreur erreur erreur.. |
| Full TF, noisy-v1 | Le modèle reprend parfois les mots quand il fait une erreur. |

## 10. If the model predicts one wrong token, the next prediction may also become wrong.

| Model | Actual output |
|---|---|
| Early 53k | Si le modèle prédit un mauvais coup, la prédiction suivante peut aussi être inaugurée. |
| Baseline 103k | Si le modèle prédit un mauvais erreur, la prochaine prévision peut aussi être mal. |
| Decay 147k (archive) | Si le modèle prédit une mauvaise fausse, la prédiction suivante peut aussi être rontée. |
| Decay 172k | Si le modèle prédit un mauvais mauvais, la, peut prévoir la la prochaine prédiction. |
| Decay best, interval 24 | Si le modèle prédit une mauvaise mauv,,, la prévision suivante peut aussi se ronter. |
| Decay 224k | Si le modèle prédit un mauvais,,, la prochaine prévision peut peut aussi être venir. |
| “No-TF” file, 228k | Si le modèle prédit un une mauv,,, la la prochaine prédiction peut aussi être être.... |
| Ten-layer continuation | Si le modèle préprédire un mauv,,,,,,, dipeut préêtre.... |
| Full TF, noisy-v1 | Si le modèle prédit une mauvaise baisse, la prochaine prévision peut aussi être erronée. |

## 11. My name is John. I am commander of the northern army. I am a general and a loyal servant to the true king.

| Model | Actual output |
|---|---|
| Early 53k | Mon nom est John. Je suis commandant de l'armée du Nord. Je suis un général et un maillon de la région. |
| Baseline 103k | Je suis le commandant de l'armée nordique. Je suis un général et un loyauté qui vont de la main. |
| Decay 147k (archive) | Mon nom est John John. Je suis commandant de l'armée nordique. Je suis un général et un loyal loyal à la véritable. |
| Decay 172k | Je suis suis le. de commandant de l'mée nordique. je suis suis un général et un loyer de la vrai. |
| Decay best, interval 24 | Je suis le, je suis, je commandant de de mée mée du Nord. je suis et et et et un loyal de la vrai. |
| Decay 224k | Mon nom est John. Je suis commandant de l ’ mée mée norNord. je suis un général et un loyoyoyoyal à la vrai. |
| “No-TF” file, 228k | Je suis est de John.. commancommandant de l mée de Nord. je suis un général et un loyoyal loyal au la vrai. |
| Ten-layer continuation | Je suis nom mon John John. je commancommande de de de de mée Nord et général et et et un oyoyoyloyoy. oy...... |
| Full TF, noisy-v1 | Mon nom est John, je suis commandant de l'armée nord, et je suis un général à part entière au midi. |

## 12. i dont know where mike is

| Model | Actual output |
|---|---|
| Early 53k | i. st sant |
| Baseline 103k | i. sait que le mouton est le |
| Decay 147k (archive) | i. t. sait où le mie est |
| Decay 172k | i sde de de où où le |
| Decay best, interval 24 | i ’ don de la ’ |
| Decay 224k | i le don de la où |
| “No-TF” file, 228k | i sde de de où le |
| Ten-layer continuation | i i de de où où le |
| Full TF, noisy-v1 | i Être saché où la fonte est |

## 13. I hav a golden aple

| Model | Actual output |
|---|---|
| Early 53k | Je suis honneur de vous dire que vous avez |
| Baseline 103k | I haho a golden |
| Decay 147k (archive) | I haha un dord |
| Decay 172k | I haa a ga dg |
| Decay best, interval 24 | I haha |
| Decay 224k | I hhhun |
| “No-TF” file, 228k | I hhd ’ un |
| Ten-layer continuation | Je hhde un |
| Full TF, noisy-v1 | Je suis un jeune homme japonais |

## 14. sixseven AI startup

| Model | Actual output |
|---|---|
| Early 53k | sixième sept heures de démarrage |
| Baseline 103k | sixième sept amendements de I |
| Decay 147k (archive) | sixsept |
| Decay 172k | sixsept de de I |
| Decay best, interval 24 | sixsept de démarrage de I |
| Decay 224k | sixsept sept de I |
| “No-TF” file, 228k | sixsept sept de I |
| Ten-layer continuation | Sxsept Adémarrage de A |
| Full TF, noisy-v1 | sept mois |

## 15. I knpw jon

| Model | Actual output |
|---|---|
| Early 53k | Je sais que je suis heureux de vous |
| Baseline 103k | I |
| Decay 147k (archive) | Je peux jj |
| Decay 172k | Je peux je |
| Decay best, interval 24 | Je pp |
| Decay 224k | I hon |
| “No-TF” file, 228k | I hp |
| Ten-layer continuation | Je h |
| Full TF, noisy-v1 | Je suis heureux de dire |

## 16. I know John, but John does not know me.

| Model | Actual output |
|---|---|
| Early 53k | Je sais John John, mais je ne sais pas. |
| Baseline 103k | Je sais John John, mais John ne me sait pas. |
| Decay 147k (archive) | Je sais John John John, mais John ne sme pas. |
| Decay 172k | Je sais John John John John John, mais sme pas. |
| Decay best, interval 24 | Je sais John John John John John John ne sspas pas. |
| Decay 224k | Je sais John John John John mais John ne sais pas. |
| “No-TF” file, 228k | Je sais John John John John mais John ne sais pas. |
| Ten-layer continuation | Je sais John John John John ne ne spas pas pas.. |
| Full TF, noisy-v1 | Je sais John John, mais je ne me sais pas. |

## 17. I have a golden apple.

| Model | Actual output |
|---|---|
| Early 53k | J'ai une pomme de cadeau. |
| Baseline 103k | J'ai une pomme d'or. |
| Decay 147k (archive) | J'ai une pomme de poudre. |
| Decay 172k | Je ai une pomme de |
| Decay best, interval 24 | Je ai une pomme de ’.. |
| Decay 224k | J'ai ai pompomme d'or. |
| “No-TF” file, 228k | Je ai ai pomme me. |
| Ten-layer continuation | J'me me me mes me mes.... |
| Full TF, noisy-v1 | J'ai une pomme germanique. |

## 18. I know John.

| Model | Actual output |
|---|---|
| Early 53k | Je sais John. |
| Baseline 103k | Je sais John John. |
| Decay 147k (archive) | Je sais John John. |
| Decay 172k | Je sais que John. |
| Decay best, interval 24 | Je sais ais John. |
| Decay 224k | Je sais John. |
| “No-TF” file, 228k | Je sais John. |
| Ten-layer continuation | Je sais ais John. |
| Full TF, noisy-v1 | Je sais John John. |

## 19. I don't know where Mike is.

| Model | Actual output |
|---|---|
| Early 53k | Je ne sais pas où Mike est. |
| Baseline 103k | Je ne sais pas où Mike est. |
| Decay 147k (archive) | Je ne sais pas où Mike est. |
| Decay 172k | Je ne sais pas où Mike est. |
| Decay best, interval 24 | Je ne sais pas Mike Mike. |
| Decay 224k | Je ne sais pas Mike est. |
| “No-TF” file, 228k | Je ne sais pas MiMike est. |
| Ten-layer continuation | Je ne sais pas où Mike est.. |
| Full TF, noisy-v1 | Je ne sais pas où Mike est. |

## 20. The price decreased from 50 euros to 30 euros.

| Model | Actual output |
|---|---|
| Early 53k | Le prix a diminué de 50 euros à 30 euros. |
| Baseline 103k | Le prix a diminué de 50 euros à 30 euros. |
| Decay 147k (archive) | Le prix a baissé de 50 euros à 30 euros. |
| Decay 172k | Le prix a baissé de 50 euros à 30 euros. |
| Decay best, interval 24 | Le prix a diminué de de 50 50 à 30 euros euros. |
| Decay 224k | Le prix a diminué de de 50 50 euros 30 30 euros. |
| “No-TF” file, 228k | Le prix a diminué de 50 50 à 30 euros. |
| Ten-layer continuation | Le prix a diminué de 50 50 à à euros 30 30 euros |
| Full TF, noisy-v1 | Le prix a diminué de 50 euros à 30 euros. |

## 21. The price increased from 30 euros to 50 euros.

| Model | Actual output |
|---|---|
| Early 53k | Le prix a augmenté de 30 euros à 50 euros. |
| Baseline 103k | Le prix a augmenté de 30 euros à 50 euros. |
| Decay 147k (archive) | Le prix a augmenté de 30 30 à 50 euros. |
| Decay 172k | Le prix a augmenté de 30 30 euros à 50 euros. |
| Decay best, interval 24 | Le prix a augmenté de 30 30 30 à euros euros euros. |
| Decay 224k | Le prix a augmenté de 30 30 à 50 euros. |
| “No-TF” file, 228k | Le prix a augmenté de 30 30 à 50 euros. |
| Ten-layer continuation | Le prix a augmenté de 30 30 30 à euros euros. |
| Full TF, noisy-v1 | Le prix a augmenté de 30 euros à 50 euros. |

## 22. I am waiting for my friend.

| Model | Actual output |
|---|---|
| Early 53k | Je vais attendre mon ami. |
| Baseline 103k | Je suis attendu à mon ami. |
| Decay 147k (archive) | Je attendrai mon ami. |
| Decay 172k | Je suis attenattend à mon ami. |
| Decay best, interval 24 | Je attenattenattenmon mon am.. |
| Decay 224k | Je suis attend mon mon ami. |
| “No-TF” file, 228k | Je suis attenattenmon amam.. |
| Ten-layer continuation | Je attenattenattenmon mon amam.. |
| Full TF, noisy-v1 | Je suis attendu pour mon ami. |

## 23. I am not waiting for my friend.

| Model | Actual output |
|---|---|
| Early 53k | Je ne vais pas attendre mon ami. |
| Baseline 103k | Je ne attends pas à mon ami. |
| Decay 147k (archive) | Je ne suis pas attenattendu mon mon ami. |
| Decay 172k | Je ne suis attenpas pas mon mon ami. |
| Decay best, interval 24 | Je ne ’ attenattenpas pas mon mon ami. |
| Decay 224k | Je ne ’ attenpas pas mon mon ami. |
| “No-TF” file, 228k | Je ne attenattenpas pas mon amam.. |
| Ten-layer continuation | Je ne vattenattenpas mon mon amam. |
| Full TF, noisy-v1 | Je ne m'attends pas à mon ami. |

## 24. The meeting starts at 9:30 tomorrow morning.

| Model | Actual output |
|---|---|
| Early 53k | La réunion commence à 9 h 30. |
| Baseline 103k | La séance débute à 9 h 30 à matin matin. |
| Decay 147k (archive) | La réunion commence à 9 h 30 matimatimatimatimatimatimatimatimatimatimatimatimatimatimatimatimatimatimatimatimatimatin. |
| Decay 172k | La réunion commence à 9 h 30 matimatimatin. |
| Decay best, interval 24 | La réunion commence à 9 h 30 matimatimatimatimati. |
| Decay 224k | La réunion commence à 9 h 30 matiain matin. |
| “No-TF” file, 228k | La réunion débute à 9 h 30 matiain matimati.. |
| Ten-layer continuation | La réunion commence à à h 30 matimatimatimati.. |
| Full TF, noisy-v1 | La séance débute à 9 h 30. |

## 25. If it rains tomorrow, we will stay at home.

| Model | Actual output |
|---|---|
| Early 53k | Si elle rationne demain, nous resterions à la maison. |
| Baseline 103k | Si elle rationalisdemain, nous resterons à la maison. |
| Decay 147k (archive) | Si elle pluvidemain, nous resterons à la maison. |
| Decay 172k | Si elle pluraain, nous resresons à la maison. |
| Decay best, interval 24 | Si il se ra, ain, nous resresons à maison maison. |
| Decay 224k | Si il rase ain ain demain, nous resresons à maison maison. |
| “No-TF” file, 228k | Si il rase ain ain, nous resresons à maison. |
| Ten-layer continuation | Si nous il rademdemdemain dem, nous resresà à.. |
| Full TF, noisy-v1 | Si elle plaît demain, nous resterons à la maison. |

## 26. Please send me the report before Friday.

| Model | Actual output |
|---|---|
| Early 53k | Veuillez envoyer le rapport avant le vendredi. |
| Baseline 103k | Veuillez vous adresser le rapport avant le vendredi. |
| Decay 147k (archive) | Veuillez vous envoyer le rapport avant le vendredi. |
| Decay 172k | Veuillez vous parvenir le rapport le rapport le vendredi. |
| Decay best, interval 24 | Veuillez vous parvenir le le le le vendredi redi. |
| Decay 224k | Veuillez vous envoyer le rapport le vendredi. |
| “No-TF” file, 228k | Veuillez envoyer le rapport le le vendredi. |
| Ten-layer continuation | Veuillez vous primoi le rapport le le redi vend.. |
| Full TF, noisy-v1 | Veuillez envoyer le rapport avant le vendredi. |

## 27. Please send me the reprot before Friday.

| Model | Actual output |
|---|---|
| Early 53k | Veuillez envoyer le reproche avant le vendredi. |
| Baseline 103k | Veuillez vous référer au nouveau vendredi. |
| Decay 147k (archive) | Veuillez vous envoyer le reproche avant le vendredi. |
| Decay 172k | Veuillez vous 'envoyer le reproavant le vendredi. |
| Decay best, interval 24 | Veuillez vous mparvenir le réavant avant le vendredi avant. vendredi. |
| Decay 224k | Veuillez 'envoyer la proproavant avant vendredi. |
| “No-TF” file, 228k | Veuillez envoyer la réproavant avant le vendredi. |
| Ten-layer continuation | Veuillez vous mz-moi la reavant avant avant redi vendvend |
| Full TF, noisy-v1 | Veuillez envoyer le dépôt avant le vendredi. |

## 28. Tom felt safe.

| Model | Actual output |
|---|---|
| Early 53k | Tom se sentait sûr. |
| Baseline 103k | Tom se trouvait sans danger. |
| Decay 147k (archive) | Tom se sentait sûr. |
| Decay 172k | Tom est sait sûr. |
| Decay best, interval 24 | Tom a sentir sûr. |
| Decay 224k | Tom a sentir sûr. |
| “No-TF” file, 228k | Tom a sentir sûr. |
| Ten-layer continuation | Tom om sait sû... |
| Full TF, noisy-v1 | Trop show show show show show show show show show show show show show show show show show show show show show show show show show show show show show show show show s |

## 29. The Commission shall comprise nine members.

| Model | Actual output |
|---|---|
| Early 53k | La Commission compose neuf membres. |
| Baseline 103k | La Commission compose neuf membres. |
| Decay 147k (archive) | La Commission comprend neuf membres. |
| Decay 172k | La Commission comprend neuf membres. |
| Decay best, interval 24 | La Commission comprend neuf membres. |
| Decay 224k | La Commission comprend neuf membres membres. |
| “No-TF” file, 228k | La Commission est composé de neuf membres. |
| Ten-layer continuation | La Commission comprend neuf membres membres. |
| Full TF, noisy-v1 | La Commission comprend neuf membres. |
